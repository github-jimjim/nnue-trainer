use std::convert::TryInto;
use std::env;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::time::Instant;
const RECORD_SIZE: usize = 40;
const MAX_DOUBLES: usize = 100_000_000;
#[derive(Clone, Copy, Debug)]
struct Record {
    sum: u64,
    offset: u64,
}
fn format_number(n: u64) -> String {
    let s = n.to_string();
    let mut parts = Vec::new();
    let mut i = s.len();
    while i > 3 {
        parts.push(&s[i - 3..i]);
        i -= 3;
    }
    parts.push(&s[..i]);
    parts.reverse();
    parts.join(".")
}
fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        println!("Usage: {} <input_file> <output_file>", args[0]);
        return Ok(());
    }
    let input_file = &args[1];
    let output_file = &args[2];
    println!("Type memory to allocate in GB: ");
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let gb: u64 = input.trim().parse().unwrap_or(1);
    let ram_bytes = gb * 1024 * 1024 * 1024;
    println!("Allocating up to {} GB of RAM...", gb);
    let file = File::open(input_file)?;
    let mut reader = BufReader::new(file);
    let file_size = reader.get_ref().metadata()?.len();
    let total_records = file_size / RECORD_SIZE as u64;
    println!("File contains {} records.", format_number(total_records));
    let max_records = ram_bytes as usize / std::mem::size_of::<Record>();
    if (total_records as usize) > max_records {
        println!(
            "Warning: RAM limit reached. Will only load {} records.",
            format_number(max_records as u64)
        );
    }
    let load_records = total_records.min(max_records as u64);
    let mut records = Vec::with_capacity(load_records as usize);
    println!("Reading file into memory...");
    let mut offset = 0u64;
    for idx in 0..load_records {
        let mut buffer = [0u8; RECORD_SIZE];
        if reader.read_exact(&mut buffer).is_err() {
            break;
        }
        let k1 = u64::from_le_bytes(buffer[0..8].try_into().unwrap());
        let k2 = u64::from_le_bytes(buffer[8..16].try_into().unwrap());
        let k3 = u64::from_le_bytes(buffer[16..24].try_into().unwrap());
        let k4 = u64::from_le_bytes(buffer[24..32].try_into().unwrap());
        records.push(Record {
            sum: k1.wrapping_add(k2).wrapping_add(k3).wrapping_add(k4),
            offset,
        });
        offset += 40;
        if idx % 10_000_000 == 0 && idx != 0 {
            println!("Read {} records...", format_number(idx));
        }
    }
    println!(
        "\nSorting {} records...",
        format_number(records.len() as u64)
    );
    let start = Instant::now();
    records.sort_unstable_by_key(|r| r.sum);
    println!("Sorting done in {:.2?}.", start.elapsed());
    println!("\nChecking for doubles...");
    let mut doubles = Vec::new();
    let mut last_sum = None;
    for record in &records {
        if let Some(prev) = last_sum {
            if prev == record.sum {
                doubles.push(record.offset);
                if doubles.len() >= MAX_DOUBLES {
                    println!("Maximum number of doubles reached!");
                    break;
                }
            }
        }
        last_sum = Some(record.sum);
    }
    println!("Found {} doubles.", format_number(doubles.len() as u64));
    println!("\nSorting doubles by offset...");
    doubles.sort_unstable();
    println!("\nCreating new file without doubles...");
    let mut reader = BufReader::new(File::open(input_file)?);
    let mut writer = BufWriter::new(File::create(output_file)?);
    let mut current_offset = 0u64;
    let mut doubles_index = 0;
    loop {
        let mut buffer = [0u8; RECORD_SIZE];
        if reader.read_exact(&mut buffer).is_err() {
            break;
        }
        if doubles_index < doubles.len() && current_offset == doubles[doubles_index] {
            doubles_index += 1;
            current_offset += 40;
            continue;
        }
        writer.write_all(&buffer)?;
        current_offset += 40;
    }
    writer.flush()?;
    println!("\nDone! Output file: {}", output_file);
    Ok(())
}
