//! Check every Gather destination against the source Spatter data and pattern.
use cyclotron::base::mem::HasMemory;
use cyclotron::ui::{make_sim, read_toml, CyclotronArgs};
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 11 {
        eprintln!("usage: spatter_full_gather_check CONFIG ELF OUTPUT_HEX LENGTH COUNT WRAP DELTA PATTERN_BIN SOURCE_BIN TIMING(0|1)");
        std::process::exit(2);
    }
    let config = PathBuf::from(&args[1]);
    let output_addr = usize::from_str_radix(args[3].trim_start_matches("0x"), 16).unwrap();
    let length: usize = args[4].parse().unwrap();
    let count: usize = args[5].parse().unwrap();
    let wrap: usize = args[6].parse().unwrap();
    let delta: usize = args[7].parse().unwrap();
    let pattern = std::fs::read(&args[8]).unwrap();
    let source = std::fs::read(&args[9]).unwrap();
    let timing = match args[10].as_str() { "0" => false, "1" => true, _ => panic!("invalid timing") };
    assert_eq!(pattern.len(), length * 4);
    let options = CyclotronArgs {
        config_path: config.clone(), binary_path: Some(PathBuf::from(&args[2])),
        timing, ..CyclotronArgs::default()
    };
    let toml = read_toml(config.as_path());
    let mut sim = make_sim(Some(&toml), &Some(options));
    sim.top.timeout = 20_000_000;
    if let Err(code) = sim.simulate() {
        eprintln!("Cyclotron failed or timed out with code {code}");
        std::process::exit(1);
    }
    let memory = sim.top.gmem.read().expect("GPU memory lock poisoned");
    let total = length * wrap;
    let output = memory.read_impl(output_addr, total * 8).expect("output out of range");
    let mut digest = 0xcbf29ce484222325u64;
    for pos in 0..total {
        let r = pos / length;
        let j = pos % length;
        let actual = u64::from_le_bytes(output[pos * 8..pos * 8 + 8].try_into().unwrap());
        let expected = if r < count {
            let base = u32::from_le_bytes(pattern[j * 4..j * 4 + 4].try_into().unwrap()) as usize;
            let last = r + ((count - 1 - r) / wrap) * wrap;
            let index = base + delta * last;
            u64::from_le_bytes(source[index * 8..index * 8 + 8].try_into().unwrap())
        } else { 0 };
        if actual != expected {
            eprintln!("Gather output[{pos}] = 0x{actual:016x}, expected 0x{expected:016x}");
            std::process::exit(1);
        }
        digest = (digest ^ (actual & 0xffff_ffff)).wrapping_mul(0x100000001b3);
        digest = (digest ^ (actual >> 32)).wrapping_mul(0x100000001b3);
    }
    for (name, addr) in [("before", output_addr - 64), ("after", output_addr + total * 8)] {
        if memory.read_impl(addr, 64).unwrap().iter().any(|&byte| byte != 0) {
            eprintln!("Gather guard {name} corrupted");
            std::process::exit(1);
        }
    }
    println!("SPATTER_FULL_GATHER_RESULT elements={total} digest={digest:016x} timing={timing}");
}
