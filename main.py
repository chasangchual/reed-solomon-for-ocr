from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def load_reed_solomon_for_ocr():
    module_path = Path(__file__).with_name("reed-solomon-ocr.py")
    spec = spec_from_file_location("reed_solomon_ocr", module_path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_path}")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ReedSolomonForOcr


ReedSolomonForOcr = load_reed_solomon_for_ocr()


if __name__ == "__main__":
    original_text = "Hello Reed-Solomon!"
    message = ReedSolomonForOcr.bytes_to_symbols(original_text.encode("utf-8"))

    # nsym = 10 can correct up to 5 unknown symbol errors.
    rs = ReedSolomonForOcr(nsym=10)

    print("Original message:")
    print(original_text)

    encoded = rs.encode(message)
    message_symbols, safe_parity = rs.encode_with_ocr_safe_parity(message)

    print("\nEncoded codeword:")
    print(encoded)

    print("\nOCR-safe parity:")
    print(safe_parity)

    print("\nIs encoded codeword valid?")
    print(rs.check(encoded))

    print("\nDoes OCR-safe parity rebuild the codeword?")
    print(rs.codeword_from_ocr_safe_parity(message_symbols, safe_parity) == encoded)

    corrupted = encoded[:]
    corrupted[0] ^= 0x55
    corrupted[5] ^= 0x33
    corrupted[10] ^= 0x77
    corrupted[15] ^= 0x22
    corrupted[20] ^= 0x11

    print("\nCorrupted codeword:")
    print(corrupted)

    print("\nIs corrupted codeword valid?")
    print(rs.check(corrupted))

    decoded = rs.correct(corrupted)
    recovered_text = ReedSolomonForOcr.symbols_to_bytes(decoded).decode("utf-8")

    print("\nRecovered message:")
    print(recovered_text)

    print("\nRecovery successful?")
    print(recovered_text == original_text)
