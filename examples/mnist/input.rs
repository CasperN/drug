use std::fs::File;
use std::io::Read;
use std::path::Path;

use byteorder::BigEndian;
use byteorder::ReadBytesExt;

static IMG_MAGIC_NUMBER: u32 = 0x0000_0803;
static LBL_MAGIC_NUMBER: u32 = 0x0000_0801;
static ROWS: usize = 28;
static COLS: usize = 28;

pub fn images(path: &Path, expected_length: u32) -> Vec<f32> {
    // Read whole file in memory
    let mut content: Vec<u8> = Vec::new();
    let mut file = {
        let mut fh = File::open(path)
            .unwrap_or_else(|_| panic!("Unable to find path to images at {:?}.", path));
        let _ = fh
            .read_to_end(&mut content)
            .unwrap_or_else(|_| panic!("Unable to read whole file in memory ({})", path.display()));
        // The read_u32() method, coming from the byteorder crate's ReadBytesExt trait, cannot be
        // used with a `Vec` directly, it requires a slice.
        &content[..]
    };

    let magic_number = file
        .read_u32::<BigEndian>()
        .unwrap_or_else(|_| panic!("Unable to read magic number from {:?}.", path));
    assert!(IMG_MAGIC_NUMBER == magic_number, "Incorrect Magic Number");

    let length = file.read_u32::<BigEndian>().unwrap() as u32;
    assert!(expected_length == length, "Unexpected Length");

    let rows = file.read_u32::<BigEndian>().unwrap() as usize;
    assert!(ROWS == rows, "Unexpected rows");

    let cols = file.read_u32::<BigEndian>().unwrap() as usize;
    assert!(COLS == cols, "Unexpected columns");

    file.to_vec()
        .into_iter()
        .map(|x| f32::from(x) / 255.0)
        .collect()
}

pub fn labels(path: &Path, expected_length: u32) -> Vec<usize> {
    let mut file =
        File::open(path).unwrap_or_else(|_| panic!("Unable to find path to labels at {:?}.", path));

    let magic_number = file
        .read_u32::<BigEndian>()
        .unwrap_or_else(|_| panic!("Unable to read magic number from {:?}.", path));
    assert!(LBL_MAGIC_NUMBER == magic_number, "Incorrect magic number");

    let length = file
        .read_u32::<BigEndian>()
        .unwrap_or_else(|_| panic!("Unable to length from {:?}.", path));

    assert!(expected_length == length, "Unexpected length");
    file.bytes().map(|b| b.unwrap() as usize).collect()
}
