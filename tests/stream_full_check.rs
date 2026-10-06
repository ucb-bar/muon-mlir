//! Compare every original-size STREAM output word with its source-suite input formula.
use cyclotron::base::mem::HasMemory;
use cyclotron::ui::{make_sim, read_toml, CyclotronArgs};
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 7 {
        eprintln!("usage: stream_full_check CONFIG ELF KIND OUTPUT_ADDRESS_HEX ELEMENTS TIMING(0|1)");
        std::process::exit(2);
    }
    let config = PathBuf::from(&args[1]);
    let output_addr = usize::from_str_radix(args[4].trim_start_matches("0x"), 16).unwrap();
    let elements: usize = args[5].parse().unwrap();
    let timing = match args[6].as_str() { "0" => false, "1" => true, _ => panic!("invalid timing") };
    let options = CyclotronArgs {
        config_path: config.clone(),
        binary_path: Some(PathBuf::from(&args[2])),
        timing,
        ..CyclotronArgs::default()
    };
    let toml = read_toml(config.as_path());
    let mut sim = make_sim(Some(&toml), &Some(options));
    sim.top.timeout = 20_000_000;
    if let Err(code) = sim.simulate() {
        eprintln!("Cyclotron failed or timed out with code {code}");
        std::process::exit(1);
    }
    let memory = sim.top.gmem.read().expect("GPU memory lock poisoned");
    let bytes = memory.read_impl(output_addr, elements * 4).expect("output out of range");
    let mut digest = 0xcbf29ce484222325u64;
    for i in 0..elements {
        let a = (i % 1024 + 1) as f32;
        let b = ((3 * i) % 1024 + 2) as f32;
        let c = ((5 * i) % 1024 + 3) as f32;
        let expected = match args[3].as_str() {
            "copy" => a,
            "scale" => 2.0 * c,
            "add" => a + b,
            "triad" => b + 2.0 * c,
            _ => panic!("invalid STREAM kind"),
        };
        let actual = u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
        if actual != expected.to_bits() {
            eprintln!("STREAM {} output[{i}] = 0x{actual:08x}, expected 0x{:08x}", args[3], expected.to_bits());
            std::process::exit(1);
        }
        digest = (digest ^ u64::from(actual)).wrapping_mul(0x100000001b3);
    }
    for (label, addr) in [("before", output_addr - 64), ("after", output_addr + elements * 4)] {
        let guard = memory.read_impl(addr, 64).expect("guard out of range");
        if guard.iter().any(|&value| value != 0) {
            eprintln!("STREAM {} guard {label} corrupted", args[3]);
            std::process::exit(1);
        }
    }
    println!("STREAM_FULL_RESULT kind={} elements={elements} digest={digest:016x} timing={timing}", args[3]);
}
