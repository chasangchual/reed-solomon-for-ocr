from concurrent.futures import ThreadPoolExecutor
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def load_codec_module():
    module_path = Path(__file__).with_name("reed-solomon-ocr.py")
    spec = spec_from_file_location("reed_solomon_ocr", module_path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_path}")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reed_solomon_ocr = load_codec_module()
ReedSolomonForOcr = reed_solomon_ocr.ReedSolomonForOcr


def make_message(length, seed):
    return [((seed * 37) + (i * 19) + (i * i * 3)) % 256 for i in range(length)]


def make_error_positions(codeword_length, error_count, seed):
    positions = list(range(codeword_length))
    positions.sort(key=lambda position: ((position * 131) + (seed * 17)) % (codeword_length * 3 + 1))
    return positions[:error_count]


def corrupt_symbols(symbols, positions, seed):
    corrupted = list(symbols)

    for index, position in enumerate(positions):
        corrupted[position] ^= ((seed + 1) * (index + 3) * 17) % 255 + 1

    return corrupted


class ReedSolomonForOcrTest(unittest.TestCase):
    def test_encode_returns_systematic_codeword(self):
        rs = ReedSolomonForOcr(nsym=10)
        message = list(b"Hello Reed-Solomon!")

        codeword = rs.encode(message)

        self.assertEqual(codeword[: len(message)], message)
        self.assertEqual(len(codeword), len(message) + rs.nsym)
        self.assertTrue(rs.check(codeword))

    def test_corrects_demo_error_pattern(self):
        rs = ReedSolomonForOcr(nsym=10)
        message = list(b"Hello Reed-Solomon!")
        codeword = rs.encode(message)
        corrupted = corrupt_symbols(codeword, [0, 5, 10, 15, 20], seed=3)

        self.assertFalse(rs.check(corrupted))
        self.assertEqual(rs.correct(corrupted), message)

    def test_corrects_50_deterministic_error_scenarios(self):
        scenarios = []

        for index in range(50):
            nsym = [2, 4, 6, 8, 10, 12, 16, 20][index % 8]
            max_errors = nsym // 2
            error_count = index % (max_errors + 1)
            message_length = 1 + ((index * 7) % (ReedSolomonForOcr.FIELD_MAX - nsym))
            scenarios.append((index, nsym, message_length, error_count))

        self.assertEqual(len(scenarios), 50)

        for seed, nsym, message_length, error_count in scenarios:
            with self.subTest(seed=seed, nsym=nsym, message_length=message_length, error_count=error_count):
                rs = ReedSolomonForOcr(nsym=nsym)
                message = make_message(message_length, seed)
                codeword = rs.encode(message)
                positions = make_error_positions(len(codeword), error_count, seed)
                corrupted = corrupt_symbols(codeword, positions, seed)

                self.assertEqual(rs.correct(corrupted), message)

    def test_ocr_safe_parity_uses_only_safe_alphabet(self):
        rs = ReedSolomonForOcr(nsym=10)
        message = list(b"OCR safe parity")

        _, safe_parity = rs.encode_with_ocr_safe_parity(message)

        self.assertEqual(len(safe_parity), rs.nsym * 2)
        self.assertLessEqual(set(safe_parity), set(rs.OCR_SAFE_ALPHABET))

    def test_ocr_safe_parity_round_trips_to_codeword(self):
        rs = ReedSolomonForOcr(nsym=10)
        message = list(b"OCR safe parity")

        codeword = rs.encode(message)
        message_symbols, safe_parity = rs.encode_with_ocr_safe_parity(message)

        self.assertEqual(message_symbols, message)
        self.assertEqual(rs.codeword_from_ocr_safe_parity(message_symbols, safe_parity), codeword)

    def test_corrects_message_with_ocr_safe_parity(self):
        rs = ReedSolomonForOcr(nsym=10)
        message = list(b"OCR safe parity")
        message_symbols, safe_parity = rs.encode_with_ocr_safe_parity(message)
        corrupted_message = corrupt_symbols(message_symbols, [0, 4, 8], seed=9)

        self.assertEqual(rs.correct_with_ocr_safe_parity(corrupted_message, safe_parity), message)

    def test_rejects_unsupported_ocr_safe_character(self):
        rs = ReedSolomonForOcr(nsym=1)

        with self.assertRaises(ValueError):
            rs.ocr_safe_to_parity("0A")

    def test_rejects_invalid_ocr_safe_symbol_pair(self):
        rs = ReedSolomonForOcr(nsym=1)

        with self.assertRaises(ValueError):
            rs.ocr_safe_to_parity("YY")

    def test_rejects_wrong_ocr_safe_parity_length(self):
        rs = ReedSolomonForOcr(nsym=3)

        with self.assertRaises(ValueError):
            rs.ocr_safe_to_parity("234")

    def test_rejects_wrong_raw_parity_length(self):
        rs = ReedSolomonForOcr(nsym=3)

        with self.assertRaises(ValueError):
            rs.parity_to_ocr_safe([1, 2])

    def test_rejects_message_that_exceeds_field_limit(self):
        rs = ReedSolomonForOcr(nsym=10)

        with self.assertRaises(ValueError):
            rs.encode([0] * 246)

    def test_bytes_symbol_helpers_round_trip(self):
        data = b"safe OCR ID 239A"

        symbols = ReedSolomonForOcr.bytes_to_symbols(data)

        self.assertEqual(ReedSolomonForOcr.symbols_to_bytes(symbols), data)

    def test_module_level_wrappers_match_class_api(self):
        message = list(b"wrapper check")
        nsym = 10
        rs = ReedSolomonForOcr(nsym=nsym)

        class_codeword = rs.encode(message)
        wrapper_codeword = reed_solomon_ocr.rs_encode_msg(message, nsym)

        self.assertEqual(wrapper_codeword, class_codeword)
        self.assertTrue(reed_solomon_ocr.rs_check(wrapper_codeword, nsym))
        self.assertEqual(reed_solomon_ocr.rs_correct_msg(wrapper_codeword, nsym), message)

    def test_shared_instance_is_safe_for_concurrent_use(self):
        rs = ReedSolomonForOcr(nsym=10)

        def round_trip(seed):
            message = make_message(48, seed)
            codeword = rs.encode(message)
            positions = make_error_positions(len(codeword), 5, seed)
            corrupted = corrupt_symbols(codeword, positions, seed)
            return rs.correct(corrupted) == message

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(round_trip, range(80)))

        self.assertTrue(all(results))

    def test_instance_state_is_immutable_after_initialization(self):
        rs = ReedSolomonForOcr(nsym=10)

        with self.assertRaises(AttributeError):
            rs.nsym = 12

        with self.assertRaises(TypeError):
            rs.gf_exp[0] = 99

    def test_constructor_rejects_invalid_parameters(self):
        with self.assertRaises(TypeError):
            ReedSolomonForOcr(nsym="10")

        with self.assertRaises(TypeError):
            ReedSolomonForOcr(nsym=True)

        with self.assertRaises(ValueError):
            ReedSolomonForOcr(nsym=0)

        with self.assertRaises(ValueError):
            ReedSolomonForOcr(nsym=255)

        with self.assertRaises(TypeError):
            ReedSolomonForOcr(nsym=10, primitive_poly="0x11d")

    def test_rejects_invalid_message_symbols(self):
        rs = ReedSolomonForOcr(nsym=10)

        with self.assertRaises(TypeError):
            rs.encode([1, "2", 3])

        with self.assertRaises(TypeError):
            rs.encode([1, True, 3])

        with self.assertRaises(ValueError):
            rs.encode([1, 256, 3])

        with self.assertRaises(TypeError):
            rs.encode(None)

    def test_rejects_invalid_codeword_symbols(self):
        rs = ReedSolomonForOcr(nsym=10)

        with self.assertRaises(ValueError):
            rs.correct([0] * 9)

        with self.assertRaises(TypeError):
            rs.correct([0] * 10 + ["bad"])

        with self.assertRaises(ValueError):
            rs.correct([0] * 256)

    def test_rejects_invalid_gf_arithmetic_inputs(self):
        rs = ReedSolomonForOcr(nsym=10)

        with self.assertRaises(TypeError):
            rs.gf_mul("1", 2)

        with self.assertRaises(ValueError):
            rs.gf_mul(256, 2)

        with self.assertRaises(ZeroDivisionError):
            rs.gf_div(1, 0)

        with self.assertRaises(TypeError):
            rs.gf_pow(2, "3")

        with self.assertRaises(ZeroDivisionError):
            rs.gf_inverse(0)

    def test_rejects_invalid_polynomial_inputs(self):
        rs = ReedSolomonForOcr(nsym=10)

        with self.assertRaises(ValueError):
            rs.poly_eval([], 2)

        with self.assertRaises(TypeError):
            rs.poly_add([1, 2], ["x"])

        with self.assertRaises(ValueError):
            rs.poly_div([1], [1, 2])

        with self.assertRaises(ZeroDivisionError):
            rs.poly_div([1, 2, 3], [0, 0])

    def test_rejects_invalid_error_metadata(self):
        rs = ReedSolomonForOcr(nsym=10)
        codeword = rs.encode(list(b"metadata"))
        syndromes = rs.calc_syndromes(codeword)

        with self.assertRaises(ValueError):
            rs.find_error_locator([0] * 10)

        with self.assertRaises(ValueError):
            rs.find_errors([1, 1], 0)

        with self.assertRaises(ValueError):
            rs.correct_errata(codeword, syndromes, [0, 0])

        with self.assertRaises(ValueError):
            rs.correct_errata(codeword, syndromes, [len(codeword)])

    def test_rejects_invalid_ocr_safe_parity_type(self):
        rs = ReedSolomonForOcr(nsym=1)

        with self.assertRaises(TypeError):
            rs.ocr_safe_to_parity(["2", "3"])

    def test_rejects_ocr_safe_message_that_exceeds_field_limit(self):
        rs = ReedSolomonForOcr(nsym=10)
        _, safe_parity = rs.encode_with_ocr_safe_parity([0] * 245)

        with self.assertRaises(ValueError):
            rs.codeword_from_ocr_safe_parity([0] * 246, safe_parity)

    def test_bytes_to_symbols_rejects_non_bytes(self):
        with self.assertRaises(TypeError):
            ReedSolomonForOcr.bytes_to_symbols("not bytes")


if __name__ == "__main__":
    unittest.main()
