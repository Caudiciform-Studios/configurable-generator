use std::{
    path::{Path, PathBuf},
    collections::{HashSet, HashMap},
};

use anyhow::{Result, Context};
use serde::{Serialize, Deserialize};
use rand::prelude::*;
use noise::{Fbm, Perlin, NoiseFn, ScalePoint};

mod value;
pub use value::Value;

pub struct Seed(SmallRng);

impl Seed {
    pub fn new(seed: u64) -> Self {
        Self(SmallRng::seed_from_u64(seed))
    }

    pub fn fork(&mut self) -> Self {
        Self(SmallRng::from_rng(&mut self.0).unwrap())
    }

    pub fn to_u64(mut self) -> u64 {
        self.0.gen()
    }

    pub fn to_rng(self) -> SmallRng {
        self.0
    }
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

    pub fn generate(&self, width: usize, height: usize, ctx: &mut Ctx) -> TileMap {
        let mut tilemap = TileMap::new(width, height);


        for modifier in &self.modifiers {
            modifier.apply(&mut tilemap, ctx);
        }

        tilemap
    }

    pub fn solidify(&mut self, rng: &mut impl Rng ) -> Ctx {
        let mut rng = SmallRng::from_rng(rng).unwrap();

        let noise = Box::new(ScalePoint::new(Fbm::<Perlin>::new(rng.gen::<u32>())).set_x_scale(0.0923).set_y_scale(0.0923));
        let mut ctx = Ctx {
            rng,
            noise,
            string_table: Default::default(),
            reverse_string_table: Default::default(),
            current_activations: 0,
        };

        if let Some((w, h)) = &mut self.default_size {
            w.solidify(&mut ctx);
            h.solidify(&mut ctx);
        }

        for modifier in &mut self.modifiers {
            modifier.solidify(&mut ctx);
        }

        ctx
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
    Packed(u32)
}

impl StringId {
    fn solidify(&mut self, ctx: &mut Ctx) {
        if let StringId::Display(s) = self {
            *self = StringId::Packed(ctx.tile_name_to_u32(s));
        }
    }

    fn as_packed(&self) -> u32 {
        match self {
            Self::Display(_) => u32::MAX,
            Self::Packed(v) => *v,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct TileMap {
    pub height: usize,
    pub width: usize,
    pub tiles: Vec<u32>,
}

impl TileMap {
    fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            tiles: vec![u32::MAX; width*height],
        }
    }

    pub fn tiles_mut(&mut self) -> impl Iterator<Item = ((u32, u32), &mut u32)> {
        self.tiles.iter_mut().enumerate().map(|(i,t)| {
            let x = (i % self.width) as u32;
            let y = (i / self.width) as u32;
            ((x,y), t)
        })
    }

    pub fn tiles(&self) -> impl Iterator<Item = ((u32, u32), &u32)> {
        self.tiles.iter().enumerate().map(|(i,t)| {
            let x = (i % self.width) as u32;
            let y = (i / self.width) as u32;
            ((x,y), t)
        })
    }
}

pub struct Ctx {
    pub rng: SmallRng,
    noise: Box<dyn NoiseFn<f64, 2>>,
    pub string_table: HashMap<String, u32>,
    pub reverse_string_table: HashMap<u32, String>,
    current_activations: u32,
}

impl Ctx {
    pub fn fork_rng(&mut self) -> SmallRng {
        SmallRng::from_rng(&mut self.rng).unwrap()
    }

    pub fn tile_name_to_u32(&mut self, ty: &str) -> u32 {
        if let Some(id) = self.string_table.get(ty) {
            *id
        } else {
            let id = self.string_table.len() as u32;
            self.string_table.insert(ty.to_string(), id);
            self.reverse_string_table.insert(id, ty.to_string());
            id
        }
    }

    pub fn u32_to_tile_name(&mut self, ty: u32) -> &str {
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
}

#[derive(Clone, Serialize, Deserialize)]
pub struct CommonParams {
    #[serde(default="tile_prob_default")]
    tile_prob: Value,
    #[serde(default="prob_default")]
    prob: Value,
    #[serde(default="noise_threshold_default")]
    noise_threshold: Value,
    #[serde(default="iterations_default")]
    iterations: Value,
    #[serde(default)]
    max_activations: Option<Value>,
    #[serde(default)]
    skip_ty: Option<TileType>,
    #[serde(default)]
    only_on_ty: Option<TileType>,
}
fn prob_default() -> Value { Value::Const(1.0) }
fn tile_prob_default() -> Value { Value::Const(1.0) }
fn noise_threshold_default() -> Value { Value::Const(f64::NEG_INFINITY) }
fn iterations_default() -> Value { Value::Const(1.0) }

impl CommonParams {
    fn skip_modifier(&self, ctx: &mut Ctx) -> bool {
        self.prob.val(&mut ctx.rng) < 1.0 && ctx.rng.gen::<f64>() > self.prob.val(&mut ctx.rng)
    }

    fn skip_tile(&self, x: u32, y: u32, tilemap: &TileMap, ctx: &mut Ctx) -> bool {
        if let Some(max) = &self.max_activations {
            if ctx.current_activations as f64 >= max.val(&mut ctx.rng) {
                return true
            }
        }
        if let Some(ty) = &self.skip_ty {
            let i = y as usize * tilemap.width + x as usize;
            if tilemap.tiles[i] == ty.as_packed() {
                return true
            }
        }
        if let Some(ty) = &self.only_on_ty {
            let i = y as usize * tilemap.width + x as usize;
            if tilemap.tiles[i] != ty.as_packed() {
                return true
            }
        }
        if self.tile_prob.val(&mut ctx.rng) < 1.0 && ctx.rng.gen::<f64>() > self.tile_prob.val(&mut ctx.rng) {
            return true
        }
        if self.noise_threshold.val(&mut ctx.rng) > f64::NEG_INFINITY && ctx.noise.get([x as f64,y as f64]) < self.noise_threshold.val(&mut ctx.rng) {
            return true
        }
        false
    }

    fn solidify(&mut self, ctx: &mut Ctx) {
        self.tile_prob.solidify(ctx);
        self.prob.solidify(ctx);
        self.noise_threshold.solidify(ctx);
        self.iterations.solidify(ctx);
        if let Some(ty) = &mut self.skip_ty {
            ty.solidify(ctx);
        }
        if let Some(ty) = &mut self.only_on_ty {
            ty.solidify(ctx);
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Modifier {
    #[serde(flatten)]
    pub logic: ModifierLogic,
    #[serde(flatten)]
    pub common_params: CommonParams,
}

impl Modifier {
    fn apply(&self, tilemap: &mut TileMap, ctx: &mut Ctx) {
        if !self.common_params.skip_modifier(ctx) {
            for _ in 0..self.common_params.iterations.val(&mut ctx.rng).max(0.0) as u32 {
                self.logic.logic().apply(tilemap, &self.common_params, ctx);
            }
        }
        ctx.modifier_finished();
    }

    fn solidify(&mut self, ctx: &mut Ctx) {
        self.common_params.solidify(ctx);
        self.logic.logic_mut().solidify(ctx);
    }

    fn load(&mut self, base_path: &Path) -> Result<Vec<PathBuf>> {
        self.logic.logic_mut().load(base_path)
    }

    fn load_from_strs(&mut self, strs: &HashMap<String, String>) -> Result<Vec<String>> {
        self.logic.logic_mut().load_from_strs(strs)
    }
}

macro_rules! modifier_logic {
    ($( $x:ident),*) => {
        #[derive(Clone, Serialize, Deserialize)]
        pub enum ModifierLogic {
            $(
                $x($x),
            )*
        }
        impl ModifierLogic {
            pub fn logic(&self) -> & dyn ModifierImpl {
                match self {
                    $(
                        ModifierLogic::$x(m) => m,
                    )*
                }
            }
            pub fn logic_mut(&mut self) -> &mut dyn ModifierImpl {
                match self {
                    $(
                        ModifierLogic::$x(m) => m,
                    )*
                }
            }
        }
    }
}

modifier_logic! {
    Fill,
    Choice,
    Scatter,
    Cellular,
    Replace,
    Worm,
    External,
    Rooms,
    Grid
}

pub trait ModifierImpl {
    fn apply(&self, tilemap: &mut TileMap, common_params: &CommonParams, ctx: &mut Ctx);
    fn solidify(&mut self, _ctx: &mut Ctx) { }
    fn load(&mut self, _base_path: &Path) -> Result<Vec<PathBuf>> { Ok(vec![]) }
    fn load_from_strs(&mut self, _strs: &HashMap<String,String>) -> Result<Vec<String>> { Ok(vec![]) }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Fill(TileType);

impl ModifierImpl for Fill {
    fn apply(&self, tilemap: &mut TileMap, common_params: &CommonParams, ctx: &mut Ctx) {
        for i in 0..tilemap.tiles.len() {
            let x = i % tilemap.width;
            let y = i / tilemap.width;
            if common_params.skip_tile(x as u32,y as u32, tilemap, ctx){
                continue
            }
            tilemap.tiles[i] = self.0.as_packed();
            ctx.tile_set();
        }
    }

    fn solidify(&mut self, ctx: &mut Ctx) {
        self.0.solidify(ctx);
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Choice(Vec<Option<TileType>>);

impl ModifierImpl for Choice {
    fn apply(&self, tilemap: &mut TileMap, common_params: &CommonParams, ctx: &mut Ctx) {
        let mut rng = ctx.fork_rng();
        for i in 0..tilemap.tiles.len() {
            let x = i % tilemap.width;
            let y = i / tilemap.width;
            if common_params.skip_tile(x as u32,y as u32, tilemap, ctx){
                continue
            }
            if let Some(t) = self.0.choose(&mut rng).cloned().flatten() {
                tilemap.tiles[i] = t.as_packed();
                ctx.tile_set();
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
        let mut rng = ctx.fork_rng();
        for _ in 0..self.0.val(&mut ctx.rng).max(0.0) as u32 {
            loop {
                let idx = rng.gen_range(0..tilemap.tiles.len());
                let x = idx % tilemap.width;
                let y = idx / tilemap.width;
                if !common_params.skip_tile(x as u32, y as u32, tilemap, ctx) {
                    tilemap.tiles[idx] = self.1.as_packed();
                    ctx.tile_set();
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
    if_ty: TileType,
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
        let mut changed = Vec::with_capacity(200);
        for ((x,y), tile) in tilemap.tiles() {
            if *tile != self.if_ty.as_packed() || self.threshold.val(&mut ctx.rng) > 8.0 || common_params.skip_tile(x as u32, y as u32, tilemap, ctx) {
                continue
            }
            let x = x as i32;
            let y = y as i32;
            let mut count = 0;
            for nx in -1..2 {
                let nx = x + nx;
                if nx >= 0 && nx < tilemap.width as i32 {
                    for ny in -1..2 {
                        let ny = y + ny;
                        if ny >= 0 && ny < tilemap.height as i32 {
                            let j = ny as usize * tilemap.width + nx as usize;
                            if self.neighbor_ty.iter().any(|ty| tilemap.tiles[j] == ty.as_packed()) {
                                count += 1;
                            }
                            if !self.neighbor_not_ty.is_empty() && self.neighbor_not_ty.iter().all(|ty| tilemap.tiles[j] != ty.as_packed()) {
                                count += 1;
                            }
                        } else if !self.neighbor_not_ty.is_empty() {
                            count += 1;
                        }
                    }
                } else if !self.neighbor_not_ty.is_empty() {
                    count += 1;
                }
            }

            if count as f64 >= self.threshold.val(&mut ctx.rng) {
                let i = y as usize * tilemap.width + x as usize;
                changed.push((i, self.new_ty.clone()));
                ctx.tile_set();
            }
        }

        for (i, ty) in changed {
            tilemap.tiles[i] = ty.as_packed();
        }
    }

    fn solidify(&mut self, ctx: &mut Ctx) {
        self.threshold.solidify(ctx);
        self.if_ty.solidify(ctx);
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
pub struct Grid {
    spacing: (Value, Value),
    tile: TileType,
}

impl ModifierImpl for Grid {
    fn apply(&self, tilemap: &mut TileMap, common_params: &CommonParams, ctx: &mut Ctx) {
        for x in (0..tilemap.width).step_by(self.spacing.0.val(&mut ctx.rng).max(0.0) as usize) {
            for y in (0..tilemap.height).step_by(self.spacing.1.val(&mut ctx.rng).max(0.0) as usize) {
                if common_params.skip_tile(x as u32,y as u32, tilemap, ctx) {
                    continue
                }
                let i = y * tilemap.width + x;
                tilemap.tiles[i] = self.tile.as_packed();
                ctx.tile_set();
            }
        }
    }

    fn solidify(&mut self, ctx: &mut Ctx) {
        self.spacing.0.solidify(ctx);
        self.spacing.1.solidify(ctx);
        self.tile.solidify(ctx);
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Replace(TileType, TileType);

impl ModifierImpl for Replace {
    fn apply(&self, tilemap: &mut TileMap, common_params: &CommonParams, ctx: &mut Ctx) {
        for i in 0..tilemap.tiles.len() {
            let x = i / tilemap.width;
            let y = i % tilemap.width;
            if common_params.skip_tile(x as u32,y as u32, tilemap, ctx) {
                continue
            }
            if tilemap.tiles[i] == self.0.as_packed() {
                tilemap.tiles[i] = self.1.as_packed();
                ctx.tile_set();
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
    steering_strength: Value,
}

impl ModifierImpl for Worm {
    fn apply(&self, tilemap: &mut TileMap, common_params: &CommonParams, ctx: &mut Ctx) {
        let mut current = ctx.rng.gen_range(0..tilemap.tiles.len());
        let mut tries = 100;
        while self.starting_ty.as_packed() != tilemap.tiles[current] || common_params.skip_tile((current%tilemap.width) as u32, (current/tilemap.width) as u32, tilemap, ctx) {
            tries -= 1;
            if tries <= 0 {
                return
            }
            current = ctx.rng.gen_range(0..tilemap.tiles.len());
        }
        let mut current_x = (current % tilemap.width) as f64;
        let mut current_y = (current / tilemap.width) as f64;
        let mut len = self.len.val(&mut ctx.rng).max(0.0) as u32;
        let mut a = ctx.noise.get([current_x as f64,current_y as f64]) * std::f64::consts::TAU;
        while len > 0 && current_x > 0.0 && current_x < tilemap.width as f64 && current_y > 0.0 && current_y < tilemap.height as f64 {
            len -= 1;
            let next_x = a.cos() * 1.0 + current_x;
            let next_y = a.sin() * 1.0 + current_y;
            if next_x > 0.0 && next_x < tilemap.width as f64 && next_y > 0.0 && next_y < tilemap.height as f64 && self.impassable.iter().find(|ty| ty.as_packed() == tilemap.tiles[next_y as usize * tilemap.width + next_x as usize]).is_none() {
                current_x = next_x;
                current_y = next_y;
            }
            tilemap.tiles[current_y as usize * tilemap.width + current_x as usize] = self.trail.as_packed();
            ctx.tile_set();
            a += ctx.noise.get([current_x + len as f64,current_y + len as f64]) * std::f64::consts::TAU * self.steering_strength.val(&mut ctx.rng);
        }
    }

    fn solidify(&mut self, ctx: &mut Ctx) {
        self.steering_strength.solidify(ctx);
        self.len.solidify(ctx);
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
                        if !common_params.skip_tile(tx,ty, tilemap, ctx) {
                            let i = ty as usize * tilemap.width + tx as usize;
                            if tx == x || ty == y || tx == x+width-1 || ty == y+height-1 {
                                if tilemap.tiles[i] != self.floor.as_packed() {
                                    tilemap.tiles[i] = self.walls.as_packed();
                                    ctx.tile_set();
                                }
                            } else {
                                tilemap.tiles[i] = self.floor.as_packed();
                                ctx.tile_set();
                            }
                        }
                    }
                }
                break
            }
        }
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
