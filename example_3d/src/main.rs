use std::env::args;

use rand::prelude::*;

use configurable_generator::{Generator, TileMap, Ctx, Value};

fn main() {
    let config_path = args().nth(1).unwrap();
    let output_path = args().nth(2).unwrap();
    let input_seed = args().nth(3).and_then(|v| v.parse::<u64>().ok()).clone();


    let (mut generator, _watch_paths) = Generator::load::<3>(&config_path).unwrap();

    let seed = if let Some(seed) = input_seed {
        seed
    } else {
        rand::thread_rng().gen()
    };

    let mut rng = SmallRng::seed_from_u64(seed);

    let mut ctx = Ctx::<3>::new(&mut rng);
    generator.solidify(&mut ctx);
    let map = generator.generate([75,75,75], &mut ctx);

    let mut vox = vox_writer::VoxWriter::create_empty();

    let empty_id = ctx.tile_name_to_u64("empty");
    let ground_id = ctx.tile_name_to_u64("ground");
    let ground2_id = ctx.tile_name_to_u64("ground2");
    let moss_id = ctx.tile_name_to_u64("moss");

    for (p, t) in map.tiles() {
        if *t == empty_id {
            continue
        }
        let cube_color = if *t == ground_id {
            130
        } else if *t == ground2_id {
            132
        } else if *t == moss_id {
            231
        } else {
            (t) % 255 + 1
        };
        vox.add_voxel(p[0] as i32, p[1] as i32, p[2] as i32, cube_color as i32);
    }

    vox.save_to_file(output_path)
        .expect("Fail to save vox file");
}
