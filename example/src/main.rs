use std::{
    path::Path,
    collections::HashMap,
    env::args,
    sync::{Arc, mpsc::{channel, Sender}},
};

use hotwatch::{Hotwatch, Event, EventKind};
use rand::prelude::*;
use image::{imageops::{resize, FilterType}, RgbImage};
use serde::{Serialize, Deserialize};

use configurable_generator::{Generator, TileMap, Ctx};

#[derive(Serialize, Deserialize)]
struct Palette(HashMap<String, [u8;3]>);

fn main() {
    let config_path = args().nth(1).unwrap();
    let output_path = args().nth(2).unwrap();
    let input_seed = args().nth(3).and_then(|v| v.parse::<u64>().ok()).clone();

    let (reload_config_tx, reload_config_rx) = channel();
    let (reload_palette_tx, reload_palette_rx) = channel();
    let reload_config_tx = Arc::new(reload_config_tx);

    let mut hotwatch = Hotwatch::new_with_custom_delay(std::time::Duration::from_millis(10)).expect("hotwatch failed to initialize!");
    hotwatch.watch("palette.ron", move |event: Event| {
        if let EventKind::Modify(_) = event.kind {
            reload_palette_tx.send(()).unwrap();
        }
    }).expect("failed to watch file!");

    let mut watchers = vec![hotwatch];

    let (mut generator, mut watch_paths) = Generator::load(&config_path).unwrap();
    for path in &watch_paths {
        watchers.push(watch(path, reload_config_tx.clone()));
    }
    let mut palette: Palette = ron::from_str(&std::fs::read_to_string("palette.ron").unwrap()).unwrap();
    let (mut tilemap, mut ctx) = generate(&generator, input_seed);
    draw(&tilemap, &ctx, &palette).save(&output_path).unwrap();

    loop {
        std::thread::sleep(std::time::Duration::from_millis(200));
        let mut dirty = false;
        if let Ok(_) = reload_palette_rx.try_recv() {
            match ron::from_str(&std::fs::read_to_string("palette.ron").unwrap()) {
                Ok(p) => {
                    palette = p;
                    dirty = true;
                },
                Err(e) => println!("{e:?}")
            }
        }
        if let Ok(_) = reload_config_rx.try_recv() {
            match Generator::load(&config_path) {
                Ok((g, new_watch_paths)) => {
                    for p in new_watch_paths {
                        if !watch_paths.contains(&p) {
                            watchers.push(watch(&p, reload_config_tx.clone()));
                            watch_paths.push(p);
                        }
                    }
                    generator = g;
                    println!("Generating..");
                    let now = std::time::Instant::now();

                    (tilemap, ctx) = generate(&generator, input_seed);
                    println!("Done in {:.2}", now.elapsed().as_secs_f32());
                    dirty = true;
                }
                Err(e) => println!("{e:?}")
            }
        }
        if dirty {
            draw(&tilemap, &ctx, &palette).save(&output_path).unwrap();
        }
    }

}

fn watch(path: &Path, channel: Arc<Sender<()>>) -> Hotwatch {
    println!("Watching {path:?}");
    let mut hotwatch = Hotwatch::new_with_custom_delay(std::time::Duration::from_millis(10)).expect("hotwatch failed to initialize!");
    hotwatch.watch(path, move |event: Event| {
        if let EventKind::Modify(_) = event.kind {
            channel.send(()).unwrap();
        }
    }).expect("failed to watch file!");
    hotwatch
}

fn generate(generator: &Generator, input_seed: Option<u64>) -> (TileMap, Ctx) {
    let mut generator = generator.clone();
    let seed = if let Some(seed) = input_seed {
        seed
    } else {
        rand::rng().random()
    };

    let mut rng = SmallRng::seed_from_u64(seed);

    let mut ctx = generator.solidify(&mut rng);
    let size = if let Some(size) = generator.default_size {
        size
    } else {
        (300, 100)
    };
    let map = generator.generate(size.0 as usize, size.1 as usize, &mut ctx);
    (map, ctx)
}

fn draw(tilemap: &TileMap, ctx: &Ctx, palette: &Palette) -> RgbImage {
    let mut img = RgbImage::from_pixel(tilemap.width as u32, tilemap.height as u32, [255, 0, 0].into());

    for (idx, tile) in tilemap.tiles.iter().enumerate() {
        let n = ctx.reverse_string_table.get(tile).map(|s| s.as_str()).unwrap_or("undefined");
        let color = palette.0.get(n).copied().unwrap_or([255,0,0]);

        let x = idx % tilemap.width;
        let y = idx / tilemap.width;
        img.put_pixel(x as u32, y as u32, color.into());
    }

    resize(&img, tilemap.width as u32 * 8, tilemap.height as u32 * 8, FilterType::Nearest)
}
