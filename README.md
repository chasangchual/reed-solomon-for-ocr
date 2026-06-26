# Reed-Solomon for OCR

## Overview

This project provides OCR-optimized error correction for printed codes, coupons, IDs, and labels.

The goal is to print a compact code with Reed-Solomon ECC and OCR-safe parity text. When OCR misreads one or more symbols, the decoder can detect that the scanned code is invalid, identify which symbol positions are inconsistent, and correct the original message when the number of errors is within the configured Reed-Solomon limit.

This is especially useful when the print environment is not ideal. Examples include dot-matrix printers with missing pins, ribbons that are running out of ink, low-resolution printing, labels that easily collect dirt, or any workflow where printed characters can be partially damaged before OCR reads them.

This can improve OCR reliability toward 100% for controlled printed-code workflows, assuming the scan quality is good enough and the number of OCR mistakes does not exceed the correction capacity.

## Reed-Solomon Error Correction

`ReedSolomonForOcr` implements Reed-Solomon over GF(256).

- Symbol size: 8 bits
- Maximum codeword length: 255 symbols
- Encoding format: `message + parity`
- Correction capacity: up to `floor(nsym / 2)` unknown symbol errors

For example, `nsym=10` adds 10 parity symbols and can correct up to 5 unknown symbol errors.

The normal byte-oriented API works with integer symbols in the range `0..255`:

```python
from importlib.util import module_from_spec, spec_from_file_location

spec = spec_from_file_location("reed_solomon_ocr", "reed-solomon-ocr.py")
module = module_from_spec(spec)
spec.loader.exec_module(module)

ReedSolomonForOcr = module.ReedSolomonForOcr

rs = ReedSolomonForOcr(nsym=10)
message = ReedSolomonForOcr.bytes_to_symbols(b"HELLO-123")

codeword = rs.encode(message)
is_valid = rs.check(codeword)
decoded = rs.correct(codeword)

assert is_valid
assert decoded == message
```

## OCR-Safe Characters

OCR mistakes often come from look-alike characters. When reducing a character set to avoid confusion, choose the most distinctive character from each look-alike group.

Recommended alphanumeric choices:

- For `0` and `O`: keep neither. Remove both `0` and `O`.
- For `1`, `I`, and `l`: keep only the number `1`. Remove capital `I` and lowercase `l`.
- For `2` and `Z`: keep the number `2`. Remove capital `Z`.
- For `5` and `S`: keep the number `5`. Remove capital `S`.
- For `8` and `B`: keep the number `8`. Remove capital `B`.
- For `6` and `G`: keep the number `6`. Remove capital `G`.
- For `V` and `U`: keep capital `U`. Remove capital `V`.

Ultimate safe character list:

- Safe numbers: `2 3 4 5 6 7 8 9`
- Safe letters: `A C D E F H J K L M N P Q R T U W X Y`

The codec uses this OCR-safe alphabet for parity text:

```text
23456789ACDEFHJKLMNPQRTUWXY
```

Because Reed-Solomon symbols are GF(256) bytes, one parity byte cannot fit into one OCR-safe character. The implementation encodes each parity byte as two OCR-safe characters.

## How To Use

### Encode With OCR-Safe Parity

```python
rs = ReedSolomonForOcr(nsym=10)
message = ReedSolomonForOcr.bytes_to_symbols(b"HELLO-123")

message_symbols, safe_parity = rs.encode_with_ocr_safe_parity(message)

print(message_symbols)
print(safe_parity)
```

Print or store both:

- `message_symbols`: the original message symbols
- `safe_parity`: OCR-safe parity characters

### Rebuild A Codeword From OCR-Safe Parity

```python
codeword = rs.codeword_from_ocr_safe_parity(message_symbols, safe_parity)
assert rs.check(codeword)
```

### Correct A Corrupted Message With OCR-Safe Parity

```python
corrupted_message = message_symbols[:]
corrupted_message[0] ^= 0x55

decoded = rs.correct_with_ocr_safe_parity(corrupted_message, safe_parity)

assert decoded == message
```

### Byte Helpers

```python
symbols = ReedSolomonForOcr.bytes_to_symbols(b"ABC123")
data = ReedSolomonForOcr.symbols_to_bytes(symbols)
```

### Compatibility Wrapper Functions

The module also exposes wrapper functions:

```python
codeword = module.rs_encode_msg(message, nsym=10)
decoded = module.rs_correct_msg(codeword, nsym=10)

message_symbols, safe_parity = module.rs_encode_msg_with_ocr_safe_parity(message, nsym=10)
decoded = module.rs_correct_msg_with_ocr_safe_parity(message_symbols, safe_parity, nsym=10)
```

## Run Demo And Tests

```bash
python3 main.py
python3 -m unittest -v
```
