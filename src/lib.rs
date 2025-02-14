use std::{
    path::{Path, PathBuf},
    collections::{HashSet, HashMap},
    hash::{DefaultHasher, Hash, Hasher},
    sync::Mutex,
};

use once_cell::sync::Lazy;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use anyhow::{Result, Context};
use serde::{Serialize, Deserialize};
use rand::prelude::*;
use noise::{Fbm, Perlin, NoiseFn, ScalePoint};

mod value;
pub use value::Value;

static MODIFIERS: Lazy<Mutex<HashMap<String, DeserializeFn>>> = Lazy::new(|| {
    let mut m = HashMap::new();
    Mutex::new(m)
});

pub fn register_modifier<T: ModifierImpl + serde::de::DeserializeOwned + 'static>(name: &str) {
    let deserialize_fn = |deserializer: &mut dyn erased_serde::Deserializer| {
        let s: T = erased_serde::deserialize(deserializer)?;
        let boxed_trait_object: Box<dyn ModifierImpl> = Box::new(s);
        Ok(boxed_trait_object)
    };
    MODIFIERS.lock().unwrap().insert(name.to_string(), deserialize_fn);
}

pub fn register_standard_modifiers() {
    register_modifier::<Fill>("Fill");
    register_modifier::<SetTag>("SetTag");
    register_modifier::<IfTag>("IfTag");
    register_modifier::<Choice>("Choice");
    register_modifier::<Cellular>("Cellular");
    register_modifier::<FlowField>("FlowField");
    register_modifier::<Grid>("Grid");
    register_modifier::<Worm>("Worm");
    register_modifier::<Scatter>("Scatter");
    register_modifier::<Replace>("Replace");
    register_modifier::<External>("External");
    register_modifier::<Rooms>("Rooms");
    register_modifier::<FloodRegion>("FloodRegion");
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Generator {
    #[serde(default)]
    pub default_size: Option<(Value, Value)>,
    pub modifiers: Vec<Modifier>
}

impl Generator {
    pub fn load(path: impl AsRef<Path>) -> Result<(Self, Vec<PathBuf>)> {
        let path = path.as_ref();
        let mut generator: Generator = ron::from_str(&std::fs::read_to_string(path).with_context(|| format!("Loading: {path:?}"))?).with_context(|| format!("Loading: {path:?}"))?;
        let mut paths = vec![path.to_path_buf()];
        let base_path = path.parent().unwrap();
        for modifier in &mut generator.modifiers {
            paths.extend(modifier.load(&base_path)?);
        }
        Ok((generator, paths))
    }

    pub fn from_str(data: &str) -> Result<Self> {
        let generator = ron::from_str(data)?;
        Ok(generator)
    }

    pub fn load_dependencies(&mut self, files: &HashMap<String, String>) -> Result<Vec<String>> {
        let mut paths = vec![];
        for modifier in &mut self.modifiers {
            paths.extend(modifier.load_from_strs(files)?);
        }
        Ok(paths)
    }

    pub fn generate(&self, dimensions: Dimensions, ctx: &mut Ctx) -> TileMap {
        let mut tilemap = TileMap::new(dimensions);


        for modifier in &self.modifiers {
            modifier.apply(&mut tilemap, ctx);
        }

        tilemap
    }

    pub fn solidify(&mut self, ctx: &mut Ctx ) {
        if let Some((w, h)) = &mut self.default_size {
            w.solidify(ctx);
            h.solidify(ctx);
        }

        for modifier in &mut self.modifiers {
            modifier.solidify(ctx);
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Tile {
    pub ty: u32,
    #[serde(default)]
    pub tags: HashSet<u32>,
}

impl Default for Tile {
    fn default() -> Self {
        Self {
            ty: u32::MAX,
            tags: HashSet::new(),
        }
    }
}


type TileType = StringId;
type Tag = StringId;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StringId {
    Display(String),
    Packed(u64)
}

impl StringId {
    fn solidify(&mut self, ctx: &mut Ctx) {
        if let StringId::Display(s) = self {
            *self = StringId::Packed(ctx.tile_name_to_u64(s));
        }
    }

    fn as_packed(&self) -> u64 {
        match self {
            Self::Display(_) => u64::MAX,
            Self::Packed(v) => *v,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct TileMapIndex(usize);

#[derive(Default, Clone, Serialize, Deserialize)]
pub struct TileMap {
    #[cfg(feature = "2d")]
    pub dimensions: [usize; 2],
    #[cfg(feature = "3d")]
    pub dimensions: [usize; 3],
    pub tiles: Vec<u64>,
    pub tile_tags: Vec<HashSet<u64>>,
    pub tags: HashSet<u64>,
}

#[cfg(feature = "2d")]
type Dimensions = [usize; 2];
#[cfg(feature = "3d")]
type Dimensions = [usize; 3];
#[cfg(feature = "2d")]
type Point = [u32; 2];
#[cfg(feature = "3d")]
type Point = [u32; 3];
#[cfg(feature = "2d")]
type PointF = [f64; 2];
#[cfg(feature = "3d")]
type PointF = [f64; 3];

impl TileMap {
    fn new(dimensions: Dimensions) -> Self {
        let len = dimensions.iter().copied().reduce(|a,b| a*b).unwrap();
        Self {
            dimensions,
            tiles: vec![u64::MAX; len],
            tile_tags: vec![HashSet::new(); len],
            tags: HashSet::new(),
        }
    }

    pub fn index_to_point(&self, idx: TileMapIndex) -> Point {
        index_to_point(idx, &self.dimensions)
    }

    pub fn point_to_index(&self, point: Point) -> Option<TileMapIndex> {
        point_to_index(point, &self.dimensions)
    }

    pub fn tiles_mut(&mut self) -> impl Iterator<Item = (Point, &mut u64)> {
        let TileMap {
            tiles,
            dimensions,
            ..
        } = self;
        tiles.iter_mut().enumerate().map(|(i,t)| {
            (index_to_point(TileMapIndex(i), dimensions), t)
        })
    }

    pub fn tiles(&self) -> impl Iterator<Item = (Point, &u64)> {
        self.tiles.iter().enumerate().map(|(i,t)| {
            (self.index_to_point(TileMapIndex(i)), t)
        })
    }

    pub fn set_tile(&mut self, point: Point, ty: &TileType, ctx: &mut Ctx, common_params: &CommonParams, tags: &[Tag]) -> bool {
        if let Some(i) = self.point_to_index(point) {
            self.set_tile_by_idx(i, ty, ctx, common_params, tags)
        } else {
            false
        }
    }

    pub fn set_tile_by_idx(&mut self, idx: TileMapIndex, ty: &TileType, ctx: &mut Ctx, common_params: &CommonParams, tags: &[Tag]) -> bool {
        let point = self.index_to_point(idx);
        if !common_params.skip_tile(point, self, ctx) {
            self.tiles[idx.0] = ty.as_packed();
            self.tile_tags[idx.0].extend(tags.iter().map(|t| t.as_packed()));
            self.tile_tags[idx.0].extend(common_params.tile_tags.iter().map(|t| t.as_packed()));
            ctx.tile_set();
            true
        } else {
            false
        }
    }

    pub fn get_tile(&self, point: Point) -> Option<u64> {
        self.point_to_index(point).and_then(|i| Some(self.get_tile_by_idx(i)))
    }

    pub fn get_tile_by_idx(&self, idx: TileMapIndex) -> u64 {
        self.tiles[idx.0]
    }

    pub fn indexes(&self) -> impl Iterator<Item = TileMapIndex> {
        (0..self.tiles.len()).into_iter().map(|i| TileMapIndex(i))
    }

    pub fn points(&self) -> impl Iterator<Item = Point> {
        let dimensions = self.dimensions;
        (0..self.tiles.len()).into_iter().map(move |i| index_to_point(TileMapIndex(i), &dimensions))
    }

    pub fn neighboors(&self, idx: TileMapIndex, radius: u32, points: &mut Vec<TileMapIndex>) {
        points.clear();
        points.push(idx);
        self.neighboors_at_dimension(0, idx, radius, points);
    }

    fn neighboors_at_dimension(&self, n: usize, idx: TileMapIndex, radius: u32, points: &mut Vec<TileMapIndex>) {
        #[cfg(feature="2d")]
        if n >= 2 {
            return
        }
        #[cfg(feature="3d")]
        if n >= 3 {
            return
        }
        let p = self.index_to_point(idx);
        for d in -(radius as i32)..radius as i32+1 {
            let v = p[n] as i32 + d;
            if v >= 0 {
                let mut pp = p;
                pp[n] = v as u32;
                if let Some(j) = self.point_to_index(pp) {
                    points.push(j);
                    self.neighboors_at_dimension(n+1, j, radius, points);
                }
            }
        }
    }

}


pub struct Ctx {
    pub rng: SmallRng,
    #[cfg(feature="2d")]
    noise: Box<dyn NoiseFn<f64, 2>>,
    #[cfg(feature="3d")]
    noise: Box<dyn NoiseFn<f64, 3>>,
    pub string_table: HashMap<String, u64>,
    pub reverse_string_table: HashMap<u64, String>,
    pub offset: Point,
    current_activations: u32,
}

fn index_to_point(idx: TileMapIndex, dimensions: &Dimensions) -> Point {
    let mut idx = idx.0 as u32;
    let mut r = Point::default();
    #[cfg(feature = "2d")]
    {
        r[0] = idx % dimensions[0] as u32;
        r[1] = idx / dimensions[0] as u32;
    }
    #[cfg(feature = "3d")]
    {
        r[0] = idx % dimensions[0] as u32;
        r[1] = ((idx - r[0])/dimensions[0] as u32) % dimensions[1] as u32;
        r[2] = (idx-r[0] -dimensions[1] as u32*r[1])/(dimensions[0] as u32*dimensions[1] as u32);
    }
    r
}

fn point_to_index(point: Point, dimensions: &Dimensions) -> Option<TileMapIndex> {
    for (v,d) in point.iter().zip(dimensions.iter()) {
        if *v as usize>= *d {
            return None
        }
    }
    #[cfg(feature = "2d")]
    return Some(TileMapIndex(point[1] as usize * dimensions[0] + point[0] as usize));
    #[cfg(feature = "3d")]
    return Some(TileMapIndex(point[2] as usize * dimensions[0] * dimensions[1] + point[1] as usize * dimensions[0] + point[0] as usize));
}

impl Ctx {
    pub fn new(rng: &mut impl Rng) -> Self {
        let mut rng = SmallRng::from_rng(rng).unwrap();

        let noise = Box::new(ScalePoint::new(Fbm::<Perlin>::new(rng.gen::<u32>())).set_x_scale(0.0923).set_y_scale(0.0923));
        Ctx {
            rng,
            noise,
            string_table: Default::default(),
            reverse_string_table: Default::default(),
            current_activations: 0,
            offset: Point::default(),
        }
    }
}


impl Ctx {
    pub fn fork_rng(&mut self) -> SmallRng {
        SmallRng::from_rng(&mut self.rng).unwrap()
    }

    pub fn tile_name_to_u64(&mut self, ty: &str) -> u64 {
        if let Some(id) = self.string_table.get(ty) {
            *id
        } else {
            let mut s = DefaultHasher::new();
            ty.hash(&mut s);
            let id = s.finish();
            self.string_table.insert(ty.to_string(), id);
            self.reverse_string_table.insert(id, ty.to_string());
            id
        }
    }

    pub fn u64_to_tile_name(&mut self, ty: u64) -> &str {
        if let Some(name) = self.reverse_string_table.get(&ty) {
            name
        } else {
            "undefined"
        }
    }

    fn tile_set(&mut self) {
        self.current_activations += 1;
    }

    fn modifier_finished(&mut self) {
        self.current_activations = 0;
    }

    fn sample_noise(&self, mut p: Point, scale: PointF) -> f64 {
        for n in 0..p.len() {
            p[n] += self.offset[n];
        }
        let mut p = p.map(|v| v as f64 );
        for n in 0..p.len() {
            p[n] *= scale[n];
        }
        self.noise.get(p)
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct CommonParams {
    #[serde(default="tile_prob_default")]
    tile_prob: Value,
    #[serde(default)]
    tile_tags: Vec<Tag>,
    #[serde(default="prob_default")]
    prob: Value,
    #[serde(default="noise_threshold_default")]
    noise_threshold: Value,
    #[serde(default="value_one")]
    noise_scale: Value,
    #[serde(default="iterations_default")]
    iterations: Value,
    #[serde(default)]
    max_activations: Option<Value>,
    #[serde(default)]
    skip_ty: Option<TileType>,
    #[serde(default)]
    only_on_ty: Option<TileType>,
    #[serde(default)]
    if_level_tag: Option<Tag>,
    #[serde(default)]
    if_not_level_tag: Option<Tag>,

    #[serde(default)]
    #[cfg(feature="2d")]
    min: Option<[Value; 2]>,
    #[cfg(feature="3d")]
    min: Option<[Value; 3]>,
    #[serde(default)]
    #[cfg(feature="2d")]
    max: Option<[Value; 2]>,
    #[cfg(feature="3d")]
    max: Option<[Value; 3]>,
}
fn prob_default() -> Value { Value::Const(1.0) }
fn tile_prob_default() -> Value { Value::Const(1.0) }
fn noise_threshold_default() -> Value { Value::Const(f64::NEG_INFINITY) }
fn iterations_default() -> Value { Value::Const(1.0) }

impl CommonParams {
    fn skip_modifier(&self, ctx: &mut Ctx, tilemap: &TileMap) -> bool {
        if let Some(tag) = &self.if_level_tag {
            if !tilemap.tags.contains(&tag.as_packed()) {
               return true
            }
        }
        if let Some(tag) = &self.if_not_level_tag {
            if tilemap.tags.contains(&tag.as_packed()) {
               return true
            }
        }
        self.prob.val(&mut ctx.rng) < 1.0 && ctx.rng.gen::<f64>() > self.prob.val(&mut ctx.rng)
    }

    fn skip_tile(&self, point: Point, tilemap: &TileMap, ctx: &mut Ctx) -> bool {
        if let Some(vs) = &self.min {
            for (v,p) in vs.iter().zip(point.iter()) {
                if *p < v.val(&mut ctx.rng).max(0.0) as u32 {
                    return true
                }
            }
        }
        if let Some(vs) = &self.max {
            for (v,p) in vs.iter().zip(point.iter()) {
                if *p > v.val(&mut ctx.rng).max(0.0) as u32 {
                    return true
                }
            }
        }
        if let Some(idx) = tilemap.point_to_index(point) {
            if let Some(max) = &self.max_activations {
                if ctx.current_activations as f64 >= max.val(&mut ctx.rng) {
                    return true
                }
            }
            if let Some(ty) = &self.skip_ty {
                if tilemap.tiles[idx.0] == ty.as_packed() {
                    return true
                }
            }
            if let Some(ty) = &self.only_on_ty {
                if tilemap.tiles[idx.0] != ty.as_packed() {
                    return true
                }
            }
            if self.tile_prob.val(&mut ctx.rng) < 1.0 && ctx.rng.gen::<f64>() > self.tile_prob.val(&mut ctx.rng) {
                return true
            }
            #[cfg(feature="2d")]
            let scale = [self.noise_scale.val(&mut ctx.rng); 2];
            #[cfg(feature="3d")]
            let scale = [self.noise_scale.val(&mut ctx.rng); 3];
            if self.noise_threshold.val(&mut ctx.rng) > f64::NEG_INFINITY && ctx.sample_noise(point, scale) < self.noise_threshold.val(&mut ctx.rng) {
                return true
            }
        }
        false
    }

    fn solidify(&mut self, ctx: &mut Ctx) {
        self.tile_prob.solidify(ctx);
        self.prob.solidify(ctx);
        self.noise_threshold.solidify(ctx);
        self.noise_scale.solidify(ctx);
        self.iterations.solidify(ctx);
        if let Some(ty) = &mut self.skip_ty {
            ty.solidify(ctx);
        }
        if let Some(ty) = &mut self.only_on_ty {
            ty.solidify(ctx);
        }
        if let Some(ty) = &mut self.if_level_tag {
            ty.solidify(ctx);
        }
        if let Some(ty) = &mut self.if_not_level_tag {
            ty.solidify(ctx);
        }
        for tag in &mut self.tile_tags {
            tag.solidify(ctx);
        }

        for v in [&mut self.min, &mut self.max] {
            if let Some(vs) = v {
                for v in vs {
                    v.solidify(ctx);
                }
            }
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Modifier {
    #[serde(flatten)]
    pub logic: Box<dyn ModifierImpl>,
    #[serde(flatten)]
    pub common_params: CommonParams,
}

impl Modifier {
    fn apply(&self, tilemap: &mut TileMap, ctx: &mut Ctx) {
        if !self.common_params.skip_modifier(ctx, tilemap) {
            for _ in 0..self.common_params.iterations.val(&mut ctx.rng).max(0.0) as u32 {
                self.logic.apply(tilemap, &self.common_params, ctx);
            }
        }
        ctx.modifier_finished();
    }

    fn solidify(&mut self, ctx: &mut Ctx) {
        self.common_params.solidify(ctx);
        self.logic.solidify(ctx);
    }

    fn load(&mut self, base_path: &Path) -> Result<Vec<PathBuf>> {
        self.logic.load(base_path)
    }

    fn load_from_strs(&mut self, strs: &HashMap<String, String>) -> Result<Vec<String>> {
        self.logic.load_from_strs(strs)
    }
}


pub trait ModifierImpl: erased_serde::Serialize + dyn_clone::DynClone+Send+Sync {
    fn apply(&self, tilemap: &mut TileMap, common_params: &CommonParams, ctx: &mut Ctx);
    fn solidify(&mut self, _ctx: &mut Ctx) { }
    fn load(&mut self, _base_path: &Path) -> Result<Vec<PathBuf>> { Ok(vec![]) }
    fn load_from_strs(&mut self, _strs: &HashMap<String,String>) -> Result<Vec<String>> { Ok(vec![]) }
}
dyn_clone::clone_trait_object!(ModifierImpl);
erased_serde::serialize_trait_object!(ModifierImpl);


impl<'de> serde::Deserialize<'de> for Box<dyn ModifierImpl> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let visitor = HelperVisitor;
        deserializer.deserialize_map(visitor)
    }
}

type DeserializeFn =
    fn(&mut dyn erased_serde::Deserializer) -> erased_serde::Result<Box<dyn ModifierImpl>>;


struct TypeVisitor {
    deserialize_fn: DeserializeFn,
}
impl<'de> serde::de::DeserializeSeed<'de> for TypeVisitor {
    type Value = Box<dyn ModifierImpl>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let mut erased = <dyn erased_serde::Deserializer>::erase(deserializer);
        let deserialize_fn = self.deserialize_fn;
        deserialize_fn(&mut erased).map_err(|e| serde::de::Error::custom(e))
    }
}
struct HelperVisitor;
impl<'de> serde::de::Visitor<'de> for HelperVisitor {
    type Value = Box<dyn ModifierImpl>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(formatter, "Trait object 'dyn Trait'")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: serde::de::MapAccess<'de>,
    {
        let type_info = map.next_key::<String>()?.ok_or(serde::de::Error::custom(
            "Expected externally tagged 'dyn Trait'",
        ))?;
        let deserialize_fn = *MODIFIERS.lock().unwrap().get(&type_info).ok_or(serde::de::Error::custom(
            format!("Unknown type for 'dyn Trait': {type_info}"),
        ))?;
        let boxed_trait_object = map.next_value_seed(TypeVisitor { deserialize_fn })?;
        Ok(boxed_trait_object)
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Fill(TileType);

impl ModifierImpl for Fill {
    fn apply(&self, tilemap: &mut TileMap, common_params: &CommonParams, ctx: &mut Ctx) {
        for i in tilemap.indexes() {
            tilemap.set_tile_by_idx(i, &self.0, ctx, common_params, &[]);
        }
    }

    fn solidify(&mut self, ctx: &mut Ctx) {
        self.0.solidify(ctx);
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct SetTag(Tag);

impl ModifierImpl for SetTag {
    fn apply(&self, tilemap: &mut TileMap, _common_params: &CommonParams, _ctx: &mut Ctx) {
        tilemap.tags.insert(self.0.as_packed());
    }

    fn solidify(&mut self, ctx: &mut Ctx) {
        self.0.solidify(ctx);
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct IfTag {
    tag: Tag,
    #[serde(default)]
    invert: bool,
    inner: Vec<Modifier>,
}

impl ModifierImpl for IfTag {
    fn apply(&self, tilemap: &mut TileMap, _common_params: &CommonParams, ctx: &mut Ctx) {
        let run = if tilemap.tags.contains(&self.tag.as_packed()) {
            !self.invert
        } else {
            self.invert
        };

        if run {
            for m in &self.inner {
                m.apply(tilemap, ctx);
            }
        }
    }

    fn load(&mut self, base_path: &Path) -> Result<Vec<PathBuf>> {
        let mut paths = vec![];
        for m in &mut self.inner {
            paths.extend(m.load(base_path)?);
        }
        Ok(paths)
    }

    fn load_from_strs(&mut self, strs: &HashMap<String,String>) -> Result<Vec<String>> {
        let mut paths = vec![];
        for m in &mut self.inner {
            paths.extend(m.load_from_strs(strs)?);
        }
        Ok(paths)
    }

    fn solidify(&mut self, ctx: &mut Ctx) {
        self.tag.solidify(ctx);
        for m in &mut self.inner {
            m.solidify(ctx);
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Choice(Vec<Option<TileType>>);

impl ModifierImpl for Choice {
    fn apply(&self, tilemap: &mut TileMap, common_params: &CommonParams, ctx: &mut Ctx) {
        let mut rng = ctx.fork_rng();
        for i in tilemap.indexes() {
            if let Some(t) = self.0.choose(&mut rng).map(|t| t.as_ref()).flatten() {
                tilemap.set_tile_by_idx(i, t, ctx, common_params, &[]);
            }
        }
    }

    fn solidify(&mut self, ctx: &mut Ctx) {
        for ty in &mut self.0 {
            if let Some(ty) = ty {
                ty.solidify(ctx);
            }
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Scatter(Value, TileType);

impl ModifierImpl for Scatter {
    fn apply(&self, tilemap: &mut TileMap, common_params: &CommonParams, ctx: &mut Ctx) {
        for _ in 0..self.0.val(&mut ctx.rng).max(0.0) as u32 {
            let mut tries = 100;
            loop {
                tries -= 1;
                if tries < 0 {
                    break
                }
                let idx = tilemap.indexes().choose(&mut ctx.rng).unwrap();
                if tilemap.set_tile_by_idx(idx, &self.1, ctx, common_params, &[]) {
                    break
                }
            }
        }
    }

    fn solidify(&mut self, ctx: &mut Ctx) {
        self.0.solidify(ctx);
        self.1.solidify(ctx);
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Cellular {
    #[serde(default)]
    if_ty: Option<TileType>,
    #[serde(default)]
    if_not_ty: Option<TileType>,
    #[serde(default)]
    neighbor_ty: Vec<TileType>,
    #[serde(default)]
    neighbor_not_ty: Vec<TileType>,
    new_ty: TileType,
    #[serde(default="live_threshold_default")]
    threshold: Value,
}
fn live_threshold_default() -> Value { Value::Range(0.0, 8.0) }

impl ModifierImpl for Cellular {
    fn apply(&self, tilemap: &mut TileMap, common_params: &CommonParams, ctx: &mut Ctx) {
        if self.if_ty.is_none() && self.if_not_ty.is_none() {
            return
        }

        let threshold = self.threshold.val(&mut ctx.rng) as i32;
        #[cfg(feature="2d")]
        if threshold > 8 {
            return
        }
        #[cfg(feature="3d")]
        if threshold > 27 {
            return
        }
        let mut indices:Vec<_> = tilemap.indexes().collect();
        indices.shuffle(&mut ctx.rng);
        let f = |neighboor_cache: &mut Vec<TileMapIndex>, idx| {
            let tile = tilemap.get_tile_by_idx(idx);
            let check = if let Some(ty) = &self.if_ty {
                tile != ty.as_packed()
            } else {
                let ty = self.if_not_ty.as_ref().unwrap();
                tile == ty.as_packed()
            };
            if check {
                return None;
            }
            let mut count = 0;
            let mut neighboor_count = 0;
            tilemap.neighboors(idx, 1, neighboor_cache);
            for neighboor_idx in neighboor_cache.iter() {
                neighboor_count += 1;
                let neighboor = tilemap.get_tile_by_idx(*neighboor_idx);
                if self.neighbor_ty.iter().any(|ty| neighboor == ty.as_packed()) {
                    count += 1;
                }
                if !self.neighbor_not_ty.is_empty() && self.neighbor_not_ty.iter().all(|ty| neighboor != ty.as_packed()) {
                    count += 1;
                }
            }
            if !self.neighbor_not_ty.is_empty() {
                #[cfg(feature="2d")]
                let d = 8 - neighboor_count;
                #[cfg(feature="3d")]
                let d = 27 - neighboor_count;
                count += d;
            }

            if count >= threshold {
                Some((idx, self.new_ty.clone()))
            } else {
                None
            }
        };
        #[cfg(not(feature = "parallel"))]
        let changed:Vec<_> = {
            let mut neighboors_cache = Vec::with_capacity(27);
            indices.into_iter().filter_map(|idx| f(&mut neighboors_cache, idx)).collect()
        };
        #[cfg(feature = "parallel")]
        let changed: Vec<_> = indices.into_par_iter().map_with(Vec::with_capacity(27), |neighboors_cache, idx| f(neighboors_cache, idx)).filter_map(|x| x).collect();

        for (idx, ty) in changed {
            tilemap.set_tile_by_idx(idx, &ty, ctx, common_params, &[]);
        }
    }

    fn solidify(&mut self, ctx: &mut Ctx) {
        self.threshold.solidify(ctx);
        if let Some(ty) = &mut self.if_ty {
            ty.solidify(ctx);
        }
        if let Some(ty) = &mut self.if_not_ty {
            ty.solidify(ctx);
        }
        for ty in &mut self.neighbor_ty {
            ty.solidify(ctx);
        }
        for ty in &mut self.neighbor_not_ty {
            ty.solidify(ctx);
        }
        self.new_ty.solidify(ctx);
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct FlowField {
    ty: TileType,
    #[serde(default="scale_value")]
    scale: Value,
}
fn scale_value() -> Value { Value::Const(1.0) }

impl ModifierImpl for FlowField {
    fn apply(&self, tilemap: &mut TileMap, common_params: &CommonParams, ctx: &mut Ctx) {
        for p in tilemap.points() {
            let scale = PointF::default().map(|_| self.scale.val(&mut ctx.rng));
            let a = ctx.sample_noise(p, scale) * std::f64::consts::TAU;
            let mut a_tag = Tag::Display(format!("angle_{a}"));
            a_tag.solidify(ctx);
            tilemap.set_tile(p, &self.ty, ctx, common_params, &[a_tag]);
        }
    }

    fn solidify(&mut self, ctx: &mut Ctx) {
        self.scale.solidify(ctx);
        self.ty.solidify(ctx);
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Grid {
    #[serde(default="value_one")]
    spacing_x: Value,
    #[serde(default="value_one")]
    spacing_y: Value,
    #[serde(default="value_one")]
    spacing_z: Value,
    tile: TileType,
}
fn value_one() -> Value { Value::Const(1.0) }

impl ModifierImpl for Grid {
    fn apply(&self, tilemap: &mut TileMap, common_params: &CommonParams, ctx: &mut Ctx) {
        let spacing = [
            self.spacing_x.val(&mut ctx.rng).max(0.0) as u32,
            self.spacing_y.val(&mut ctx.rng).max(0.0) as u32,
            self.spacing_z.val(&mut ctx.rng).max(0.0) as u32,
        ];
        'outer: for p in tilemap.points() {
            for n in 0..p.len() {
                if p[n] % spacing[n] != 0 {
                    continue 'outer;
                }
            }
            tilemap.set_tile(p, &self.tile, ctx, common_params, &[]);
        }
    }

    fn solidify(&mut self, ctx: &mut Ctx) {
        self.spacing_x.solidify(ctx);
        self.spacing_y.solidify(ctx);
        self.spacing_z.solidify(ctx);
        self.tile.solidify(ctx);
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Replace(TileType, TileType);

impl ModifierImpl for Replace {
    fn apply(&self, tilemap: &mut TileMap, common_params: &CommonParams, ctx: &mut Ctx) {
        let mut idxs = tilemap.indexes().collect::<Vec<_>>();
        idxs.shuffle(&mut ctx.rng);
        for i in idxs {
            if tilemap.get_tile_by_idx(i) == self.0.as_packed() {
                tilemap.set_tile_by_idx(i, &self.1, ctx, common_params, &[]);
            }
        }
    }

    fn solidify(&mut self, ctx: &mut Ctx) {
        self.0.solidify(ctx);
        self.1.solidify(ctx);
    }
}



#[derive(Clone, Serialize, Deserialize)]
pub struct Worm {
    #[serde(default)]
    impassable: Vec<TileType>,
    starting_ty: TileType,
    len: Value,
    trail: TileType,
    #[serde(default="default_worm_radius")]
    radius: Value,
    #[serde(default="value_one")]
    steering_strength_x: Value,
    #[serde(default="value_one")]
    steering_strength_y: Value,
    #[serde(default="value_one")]
    steering_strength_z: Value,
}
fn default_worm_radius() -> Value { Value::Const(1.0) }

impl ModifierImpl for Worm {
    fn apply(&self, tilemap: &mut TileMap, common_params: &CommonParams, ctx: &mut Ctx) {
        let Some((current_p, _current)) = tilemap.tiles().filter(|(_, t)| **t == self.starting_ty.as_packed()).choose(&mut ctx.rng) else { return };

        let mut current_p = current_p.map(|v| v as f64);

        let mut len = self.len.val(&mut ctx.rng).max(0.0) as u32;

        let scale = PointF::default().map(|_| 1.0);
        let mut a = Point::default().map(|_| ctx.sample_noise(current_p.map(|v| v.max(0.0) as u32), scale) * std::f64::consts::TAU);
        let radius = self.radius.val(&mut ctx.rng).max(0.0);
        let radius_squared = radius.powi(2);
        #[cfg(feature="2d")]
        let mut neighboors_cache = Vec::with_capacity(8);
        #[cfg(feature="3d")]
        let mut neighboors_cache = Vec::with_capacity(27);
        while len > 0 && current_p.iter().enumerate().all(|(i, v)| *v >= 0.0 && (*v as usize) < tilemap.dimensions[i]) {
            len -= 1;
            let mut next_p = current_p;
            #[cfg(feature="2d")]
            {
                next_p[0] += a[0].cos();
                next_p[1] += a[1].sin();
            }
            #[cfg(feature="3d")]
            {
                next_p[0] += a[0].cos()*a[1].cos();
                next_p[1] += a[0].sin()*a[1].cos();
                next_p[2] += a[1].sin();
            }
            if next_p.iter().all(|v| *v >= 0.0) {
                let next_pi = next_p.map(|v| v as u32);
                if let Some(next_idx) = tilemap.point_to_index(next_pi) {
                    if self.impassable.iter().find(|ty| Some(ty.as_packed()) == tilemap.get_tile(next_pi)).is_none() {
                        //TODO: handle 3d
                        let aa = (next_p[1]-current_p[1]).atan2(next_p[0]-current_p[0]) + std::f64::consts::FRAC_PI_2;
                        let mut a_tag = Tag::Display(format!("angle_{aa}"));
                        current_p = next_p;
                        a_tag.solidify(ctx);
                        tilemap.neighboors(next_idx, radius.ceil() as u32, &mut neighboors_cache);
                        for p in &neighboors_cache {
                            let n = tilemap.index_to_point(*p);
                            if current_p.iter().zip(n.iter()).map(|(a,b)| (*b as f64- *a).powi(2)).into_iter().sum::<f64>() <= radius_squared {
                                tilemap.set_tile_by_idx(*p, &self.trail, ctx, common_params, &[a_tag.clone()]);
                            }
                        }
                    }
                }
            }
            for n in 0..current_p.len() {
                let steer_strength = match n {
                    1 => self.steering_strength_x.val(&mut ctx.rng),
                    2 => self.steering_strength_y.val(&mut ctx.rng),
                    _ => self.steering_strength_z.val(&mut ctx.rng),
                };
                a[n] += ctx.sample_noise(current_p.map(|v| v.max(0.0) as u32 + n as u32 * 1000), scale) * std::f64::consts::TAU * steer_strength;
            }
        }
    }

    fn solidify(&mut self, ctx: &mut Ctx) {
        self.steering_strength_x.solidify(ctx);
        self.steering_strength_y.solidify(ctx);
        self.steering_strength_z.solidify(ctx);
        self.len.solidify(ctx);
        self.radius.solidify(ctx);
        self.starting_ty.solidify(ctx);
        self.trail.solidify(ctx);
        for ty in &mut self.impassable {
            ty.solidify(ctx);
        }
    }
}



#[derive(Clone, Serialize, Deserialize)]
pub struct External(
    String,
    #[serde(default)]
    Option<Vec<Modifier>>
);
impl ModifierImpl for External {
    fn apply(&self, tilemap: &mut TileMap, _common_params: &CommonParams, ctx: &mut Ctx) {
        if let Some(modifiers) = &self.1 {
            for modifier in modifiers {
                modifier.apply(tilemap, ctx);
            }
        }
    }

    fn solidify(&mut self, ctx: &mut Ctx) {
        if let Some(modifiers) = &mut self.1 {
            for modifier in modifiers {
                modifier.solidify(ctx);
            }
        }
    }

    fn load(&mut self, base_path: &Path) -> Result<Vec<PathBuf>> {
        let (generator, paths) = Generator::load(base_path.join(&self.0))?;
        self.1 = Some(generator.modifiers);
        Ok(paths)
    }

    fn load_from_strs(&mut self, strs: &HashMap<String,String>) -> Result<Vec<String>> {
        if let Some(modifiers) = &mut self.1 {
            let mut paths = vec![];
            for modifier in modifiers {
                paths.extend(modifier.load_from_strs(strs)?);
            }
            Ok(paths)
        } else {
            if let Some(data) = strs.get(&self.0) {
                let mut generator = Generator::from_str(data)?;
                let paths = generator.load_dependencies(strs)?;
                self.1 = Some(generator.modifiers);
                Ok(paths)
            } else {
                Ok(vec![self.0.clone()])
            }
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Rooms {
    count: Value,
    width: Value,
    height: Value,
    floor: TileType,
    walls: TileType,
    #[serde(default)]
    max_overlaps: Value,
}

impl ModifierImpl for Rooms {
    fn apply(&self, tilemap: &mut TileMap, common_params: &CommonParams, ctx: &mut Ctx) {
        unimplemented!()
            /*
        let mut overlaps = 0;
        let max_overlaps = self.max_overlaps.val(&mut ctx.rng).max(0.0) as u32;
        let mut rooms = vec![];
        for _ in 0..self.count.val(&mut ctx.rng).max(0.0) as u32 {
            'retry: for _ in 0..200 {
                let width = self.width.val(&mut ctx.rng).max(0.0) as u32;
                let height = self.height.val(&mut ctx.rng).max(0.0) as u32;
                let x = ctx.rng.gen_range(0..tilemap.width as u32-width);
                let y = ctx.rng.gen_range(0..tilemap.height as u32-height);
                let mut new_overlaps = 0;
                for (rx,ry,rw,rh) in &rooms {
                    if *rx < x+width && rx+rw >= x && *ry < y+height && ry+rh >= y {
                        if overlaps+new_overlaps >= max_overlaps {
                            continue 'retry
                        }
                        new_overlaps += 1;
                    }
                }
                rooms.push((x,y,width,height));
                overlaps += new_overlaps;
                for tx in x..x+width {
                    for ty in y..y+height {
                        let i = ty as usize * tilemap.width + tx as usize;
                        if tx == x || ty == y || tx == x+width-1 || ty == y+height-1 {
                            if tilemap.tiles[i] != self.floor.as_packed() {
                                tilemap.set_tile_by_idx(i, &self.walls, ctx, common_params, &[]);
                            }
                        } else {
                            tilemap.set_tile_by_idx(i, &self.floor, ctx, common_params, &[]);
                        }
                    }
                }
                break
            }
        }
        */
    }

    fn solidify(&mut self, ctx: &mut Ctx) {
        self.count.solidify(ctx);
        self.width.solidify(ctx);
        self.height.solidify(ctx);
        self.floor.solidify(ctx);
        self.walls.solidify(ctx);
        self.max_overlaps.solidify(ctx);
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct FloodRegion {
    with: TileType,
    bounded_by: TileType,
    #[serde(default)]
    unless_contains: Option<TileType>,
    #[serde(default)]
    if_contains: Option<TileType>,
}
impl ModifierImpl for FloodRegion {
    fn apply(&self, tilemap: &mut TileMap, common_params: &CommonParams, ctx: &mut Ctx) {
        let boundry = self.bounded_by.as_packed();
        let mut tiles = if let Some(seed) = &self.unless_contains {
            let seed = seed.as_packed();
            tilemap.tiles.iter().map(|t| if *t == boundry { u32::MAX } else if *t == seed { 1 } else { 0 } ).collect::<Vec<_>>()
        } else if let Some(seed) = &self.if_contains {
            let seed = seed.as_packed();
            tilemap.tiles.iter().map(|t| if *t == boundry { u32::MAX } else if *t == seed { 1 } else { 0 } ).collect::<Vec<_>>()
        } else {
            return
        };

        #[cfg(feature = "2d")]
        let mut neighboors_cache = Vec::with_capacity(8);
        #[cfg(feature = "3d")]
        let mut neighboors_cache = Vec::with_capacity(27);

        let mut did_work = true;
        while did_work {
            did_work = false;
            'outer: for i in tilemap.indexes() {
                if tiles[i.0] == 0 {
                    tilemap.neighboors(i, 1, &mut neighboors_cache);
                    for neighboor in &mut neighboors_cache {
                        if 0 < tiles[neighboor.0] && tiles[neighboor.0] < u32::MAX {
                            tiles[i.0] = tiles[neighboor.0];
                            did_work = true;
                            continue 'outer
                        }
                    }
                }
            }
        }

        for (i, v) in tiles.iter().enumerate() {
            if self.unless_contains.is_none() {
                if 0 < *v && tiles[i] < u32::MAX {
                    tilemap.set_tile_by_idx(TileMapIndex(i), &self.with, ctx, common_params, &[]);
                }
            } else {
                if 0  == *v {
                    tilemap.set_tile_by_idx(TileMapIndex(i), &self.with, ctx, common_params, &[]);
                }
            }
        }
    }

    fn solidify(&mut self, ctx: &mut Ctx) {
        self.with.solidify(ctx);
        self.bounded_by.solidify(ctx);
        if let Some(v) = &mut self.unless_contains {
            v.solidify(ctx);
        }
        if let Some(v) = &mut self.if_contains {
            v.solidify(ctx);
        }
    }
}
