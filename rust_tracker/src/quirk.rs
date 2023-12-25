use crate::knappen::{run_point, ATLAS_RADII_PIXEL};
use csv::{WriterBuilder, Writer};
use nalgebra::Vector4;
use serde::Deserialize;
use std::env;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::time::Instant;

#[derive(Debug, Deserialize)]
struct Particle {
    E: f64,
    px: f64,
    py: f64,
    pz: f64,
}

pub fn process(lambda: f64, input_path_p: &str, input_path_a: &str, output_path: &str) -> Vec<i32> {
    let mut rdr_p = csv::ReaderBuilder::new().has_headers(true).from_path(input_path_p).unwrap();
    let mut rdr_a = csv::ReaderBuilder::new().has_headers(true).from_path(input_path_a).unwrap();

    let particles_p: Vec<Particle> = rdr_p.deserialize().map(|result| result.unwrap()).collect();
    let particles_a: Vec<Particle> = rdr_a.deserialize().map(|result| result.unwrap()).collect();

    std::fs::create_dir_all(Path::new(output_path).parent().unwrap()).unwrap();
    let file = File::create(output_path).unwrap();
    let mut writer = BufWriter::new(file);
    let mut csv_writer = Writer::from_writer(writer);

    // Write headers
    csv_writer.write_record(&["EventID", "PID", "Layer", "r[cm]", "phi", "z[cm]", "E[GeV]", "px[GeV]", "py[GeV]", "pz[GeV]"]).unwrap();


    let mut passed = Vec::new();
    let mut buffer: Vec<Vec<String>> = Vec::new();


    for (event_id, (particle_p, particle_a)) in particles_p.iter().zip(particles_a.iter()).enumerate().take(500) {
        let vec1 = Vector4::new(particle_p.E, particle_p.px, particle_p.py, particle_p.pz);
        let vec2 = Vector4::new(particle_a.E, particle_a.px, particle_a.py, particle_a.pz);

        let vecs = [vec2, vec1];

        match run_point(&vec1, &vec2, lambda, false, true) {
            Ok(aa) => {
                if aa.len() == 16 {
                    passed.push(event_id as i32 + 1);
                    for hit in aa {
                        let record = vec![
                            (event_id + 1).to_string(),
                            hit.6.to_string(),
                            hit.0.to_string(),
                            ATLAS_RADII_PIXEL[hit.0 as usize - 1].to_string(),
                            hit.2.to_string(),
                            hit.1.to_string(),
                            vecs[hit.6 as usize - 1][0].to_string(),
                            vecs[hit.6 as usize - 1][1].to_string(),
                            vecs[hit.6 as usize - 1][2].to_string(),
                            vecs[hit.6 as usize - 1][3].to_string(),
                        ];
                        buffer.push(record);
                        if buffer.len() >= 10 {
                            // Write buffered records to file
                            for record in buffer.drain(..) {
                                csv_writer.write_record(&record).unwrap();
                            }
                        }
                    }
                }
            }
            Err(e) => {
                continue;
            }
        }

        println!("Event ID: {}", event_id + 1);
        
        // layer = AA[jj][0] - 1
        //         r = ATLASradiiPixel[layer]
        //         z = AA[jj][1]
        //         phi = AA[jj][2]
        //         QID = AA[jj][6] - 1
        //         listhere = [EventID+1, layer+1, QID+1, r, phi, z, vecs[QID][0], vecs[QID][1], vecs[QID][2], vecs[QID][3]] # ensure the +1s are correct???
        //         data.append(listhere)
        
    }
    
    for record in buffer {
        csv_writer.write_record(&record).unwrap();
    }

    csv_writer.flush().unwrap();

    passed
}

pub fn _default_quirk() {
    let lambda = 20.0;
    let pid = 15;
    let mass = 500;

    let current_directory = env::current_dir().unwrap();
    let input_path_p = format!("{}/4vector_{}GeV_PID{}_1jet.csv", current_directory.display(), mass, pid);
    let input_path_a = format!("{}/4vector_{}GeV_PID{}_1jet.csv", current_directory.display(), mass, -pid);
    let output_path = format!("{}/HitFiles/QuirkMass_{}_Lambda_{}.csv", current_directory.display(), mass, lambda);

    let start_time = Instant::now();
    let passed = process(lambda, &input_path_p, &input_path_a, &output_path);
    let duration = start_time.elapsed();

    println!("(Rust Quirk Regular) Time taken: {:?} seconds", duration.as_secs_f32());
    println!("passed: {:?}", passed);
}

// (Quirk Regular) Time taken: 53.58206 seconds
// passed: [4, 5, 7, 8, 9, 10, 11, 12, 18, 20, 21, 25, 26, 27, 32, 33, 42, 44, 46, 47, 49, 50, 51, 52, 54, 56, 57, 58, 60, 70, 73, 75, 77, 79, 81, 82, 84, 85, 87, 89, 92, 95, 96, 102, 106, 107, 108, 109, 110, 112, 115, 122, 123, 126, 127, 128, 129, 131, 132, 139, 140, 147, 155, 158, 159, 160, 164, 170, 172, 173, 174, 175, 180, 182, 183, 184, 185, 188, 190, 191, 195, 197, 199, 201, 205, 206, 211, 215, 218, 220, 222, 223, 225, 226, 227, 231, 234, 235, 240, 241, 246, 248, 249, 254, 255, 257, 260, 261, 262, 266, 268, 275, 277, 278, 281, 282, 283, 284, 285, 288, 291, 294, 296, 300, 301, 306, 307, 308, 309, 310, 317, 319, 321, 333, 340, 344, 347, 350, 352, 353, 359, 360, 363, 365, 370, 371, 373, 375, 378, 380, 382, 384, 387, 388, 392, 399, 401, 403, 407, 409, 416, 423, 424, 429, 436, 437, 441, 444, 446, 449, 451, 457, 458, 459, 461, 462, 465, 467, 468, 471, 476, 481, 485, 491, 493, 497, 499, 500]
