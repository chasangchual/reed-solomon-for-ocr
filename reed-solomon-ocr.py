"""
Reed-Solomon implementation over GF(256).

- Symbol size: 8 bits
- Field: GF(2^8)
- Primitive polynomial: 0x11d
- Supports systematic RS encoding:
    message + parity
- Supports correction of unknown symbol errors.

Example:
    nsym = 10 means 10 parity symbols.
    It can correct up to floor(nsym / 2) = 5 unknown symbol errors.
"""


class ReedSolomonForOcr:
    """
    Reed-Solomon codec over GF(256).

    Args:
        nsym: Number of parity symbols.
        primitive_poly: Primitive polynomial used to build GF(256).
    """

    PRIMITIVE_POLY = 0x11D
    FIELD_SIZE = 256
    FIELD_MAX = 255
    OCR_SAFE_ALPHABET = "23456789ACDEFHJKLMNPQRTUWXY"
    __slots__ = ("nsym", "primitive_poly", "gf_exp", "gf_log", "_ocr_safe_index", "_locked")

    def __init__(self, nsym, primitive_poly=PRIMITIVE_POLY):
        object.__setattr__(self, "_locked", False)
        self._validate_nsym(nsym)
        self._validate_primitive_poly(primitive_poly)

        self.nsym = nsym
        self.primitive_poly = primitive_poly
        self.gf_exp, self.gf_log = self._init_tables(primitive_poly)
        self._ocr_safe_index = {
            char: index for index, char in enumerate(self.OCR_SAFE_ALPHABET)
        }
        object.__setattr__(self, "_locked", True)

    def __setattr__(self, name, value):
        if getattr(self, "_locked", False):
            raise AttributeError("ReedSolomonForOcr instances are immutable after initialization")

        object.__setattr__(self, name, value)

    @classmethod
    def _validate_nsym(cls, nsym):
        if not isinstance(nsym, int) or isinstance(nsym, bool):
            raise TypeError("nsym must be an integer")
        if nsym <= 0:
            raise ValueError("nsym must be positive")
        if nsym >= cls.FIELD_MAX:
            raise ValueError("nsym must be less than 255")

    @classmethod
    def _validate_primitive_poly(cls, primitive_poly):
        if not isinstance(primitive_poly, int) or isinstance(primitive_poly, bool):
            raise TypeError("primitive_poly must be an integer")
        if primitive_poly <= 0:
            raise ValueError("primitive_poly must be positive")

    @classmethod
    def _validate_symbol(cls, symbol, name="symbol"):
        if not isinstance(symbol, int) or isinstance(symbol, bool):
            raise TypeError(f"{name} must be an integer")
        if symbol < 0 or symbol > cls.FIELD_MAX:
            raise ValueError(f"{name} must be in range 0~255")

    @classmethod
    def _validate_symbols(cls, symbols, name):
        if isinstance(symbols, (bytes, bytearray)):
            return list(symbols)

        try:
            values = list(symbols)
        except TypeError as exc:
            raise TypeError(f"{name} must be an iterable of integers") from exc

        for index, symbol in enumerate(values):
            cls._validate_symbol(symbol, f"{name}[{index}]")

        return values

    def _validate_codeword(self, codeword):
        values = self._validate_symbols(codeword, "codeword")

        if len(values) < self.nsym:
            raise ValueError("Codeword must contain at least nsym parity symbols")
        if len(values) > self.FIELD_MAX:
            raise ValueError("Codeword is too long for GF(256)")

        return values

    @staticmethod
    def _validate_polynomial(poly, name):
        values = ReedSolomonForOcr._validate_symbols(poly, name)

        if not values:
            raise ValueError(f"{name} must not be empty")

        return values

    # ============================================================
    # 1. GF(256) table initialization
    # ============================================================

    def _init_tables(self, primitive_poly):
        """
        Build exponent and log tables for GF(256).

        GF multiplication/division can be expensive if calculated directly.
        So we use log/antilog tables.

        In GF(256):
            multiplication becomes addition of logarithms.
            division becomes subtraction of logarithms.
        """
        gf_exp = [0] * (self.FIELD_MAX * 2)
        gf_log = [0] * self.FIELD_SIZE

        x = 1

        for i in range(self.FIELD_MAX):
            gf_exp[i] = x
            gf_log[x] = i

            x <<= 1

            if x & self.FIELD_SIZE:
                x ^= primitive_poly

            x &= self.FIELD_MAX

        # Duplicate table to avoid modulo operation in multiplication.
        for i in range(self.FIELD_MAX, self.FIELD_MAX * 2):
            gf_exp[i] = gf_exp[i - self.FIELD_MAX]

        return tuple(gf_exp), tuple(gf_log)

    # ============================================================
    # 2. GF(256) arithmetic
    # ============================================================

    @staticmethod
    def gf_add(x, y):
        """
        Addition in GF(256).

        In binary finite fields, addition is XOR.
        """
        ReedSolomonForOcr._validate_symbol(x, "x")
        ReedSolomonForOcr._validate_symbol(y, "y")
        return x ^ y

    @staticmethod
    def gf_sub(x, y):
        """
        Subtraction in GF(256).

        Same as addition because XOR is its own inverse.
        """
        ReedSolomonForOcr._validate_symbol(x, "x")
        ReedSolomonForOcr._validate_symbol(y, "y")
        return x ^ y

    def gf_mul(self, x, y):
        """
        Multiplication in GF(256).
        """
        self._validate_symbol(x, "x")
        self._validate_symbol(y, "y")

        if x == 0 or y == 0:
            return 0

        return self.gf_exp[self.gf_log[x] + self.gf_log[y]]

    def gf_div(self, x, y):
        """
        Division in GF(256).
        """
        self._validate_symbol(x, "x")
        self._validate_symbol(y, "y")

        if y == 0:
            raise ZeroDivisionError("Division by zero in GF(256)")

        if x == 0:
            return 0

        return self.gf_exp[(self.gf_log[x] + self.FIELD_MAX - self.gf_log[y]) % self.FIELD_MAX]

    def gf_pow(self, x, power):
        """
        Power operation in GF(256).
        """
        self._validate_symbol(x, "x")
        if not isinstance(power, int) or isinstance(power, bool):
            raise TypeError("power must be an integer")

        if x == 0:
            return 0

        return self.gf_exp[(self.gf_log[x] * power) % self.FIELD_MAX]

    def gf_inverse(self, x):
        """
        Multiplicative inverse in GF(256).
        """
        self._validate_symbol(x, "x")

        if x == 0:
            raise ZeroDivisionError("Zero has no multiplicative inverse")

        return self.gf_exp[self.FIELD_MAX - self.gf_log[x]]

    # ============================================================
    # 3. Polynomial operations over GF(256)
    # ============================================================

    def poly_scale(self, poly, scalar):
        """
        Multiply polynomial by a scalar.
        """
        poly = self._validate_polynomial(poly, "poly")
        self._validate_symbol(scalar, "scalar")

        return [self.gf_mul(coef, scalar) for coef in poly]

    def poly_add(self, p, q):
        """
        Add two polynomials over GF(256).
        Addition is XOR coefficient-wise.
        """
        p = self._validate_polynomial(p, "p")
        q = self._validate_polynomial(q, "q")
        result = [0] * max(len(p), len(q))

        for i, coef in enumerate(p):
            result[i + len(result) - len(p)] ^= coef

        for i, coef in enumerate(q):
            result[i + len(result) - len(q)] ^= coef

        return result

    def poly_mul(self, p, q):
        """
        Multiply two polynomials over GF(256).
        """
        p = self._validate_polynomial(p, "p")
        q = self._validate_polynomial(q, "q")
        result = [0] * (len(p) + len(q) - 1)

        for j, q_coef in enumerate(q):
            for i, p_coef in enumerate(p):
                result[i + j] ^= self.gf_mul(p_coef, q_coef)

        return result

    def poly_eval(self, poly, x):
        """
        Evaluate polynomial at x using Horner's method.
        """
        poly = self._validate_polynomial(poly, "poly")
        self._validate_symbol(x, "x")
        y = poly[0]

        for coef in poly[1:]:
            y = self.gf_mul(y, x) ^ coef

        return y

    def poly_div(self, dividend, divisor):
        """
        Polynomial division over GF(256).

        Returns:
            quotient, remainder
        """
        dividend = self._validate_polynomial(dividend, "dividend")
        divisor = self._validate_polynomial(divisor, "divisor")
        if len(dividend) < len(divisor):
            raise ValueError("dividend must be at least as long as divisor")
        if max(divisor) == 0:
            raise ZeroDivisionError("Polynomial divisor must not be zero")

        result = list(dividend)

        for i in range(len(dividend) - len(divisor) + 1):
            coef = result[i]

            if coef != 0:
                for j in range(1, len(divisor)):
                    if divisor[j] != 0:
                        result[i + j] ^= self.gf_mul(divisor[j], coef)

        separator = -(len(divisor) - 1)

        quotient = result[:separator]
        remainder = result[separator:]

        return quotient, remainder

    # ============================================================
    # 4. Reed-Solomon generator polynomial
    # ============================================================

    def generator_poly(self):
        """
        Generate Reed-Solomon generator polynomial.
        """
        generator = [1]

        for i in range(self.nsym):
            generator = self.poly_mul(generator, [1, self.gf_pow(2, i)])

        return generator

    # ============================================================
    # 5. Reed-Solomon encoding
    # ============================================================

    def encode(self, message):
        """
        Encode message using Reed-Solomon.

        Args:
            message: list of integers, each 0~255

        Returns:
            codeword = message + parity
        """
        message = self._validate_symbols(message, "message")

        if len(message) + self.nsym > self.FIELD_MAX:
            raise ValueError("Message is too long for GF(256). Maximum codeword length is 255 symbols.")

        generator = self.generator_poly()

        # Append nsym zeros for parity space.
        msg_out = list(message) + [0] * self.nsym

        # Polynomial division.
        for i in range(len(message)):
            coef = msg_out[i]

            if coef != 0:
                for j in range(1, len(generator)):
                    msg_out[i + j] ^= self.gf_mul(generator[j], coef)

        parity = msg_out[-self.nsym:]

        return list(message) + parity

    def parity_to_ocr_safe(self, parity):
        """
        Encode parity bytes using only OCR-safe characters.

        Each GF(256) parity symbol is represented by two OCR-safe
        base-27 characters. This keeps one parity symbol localized to one
        character pair when OCR introduces errors.
        """
        safe_base = len(self.OCR_SAFE_ALPHABET)
        safe_chars = []
        parity = self._validate_symbols(parity, "parity")
        if len(parity) != self.nsym:
            raise ValueError(f"parity must contain exactly {self.nsym} symbols")

        for symbol in parity:
            safe_chars.append(self.OCR_SAFE_ALPHABET[symbol // safe_base])
            safe_chars.append(self.OCR_SAFE_ALPHABET[symbol % safe_base])

        return "".join(safe_chars)

    def ocr_safe_to_parity(self, safe_parity):
        """
        Decode OCR-safe parity characters back into GF(256) parity symbols.
        """
        if not isinstance(safe_parity, str):
            raise TypeError("safe_parity must be a string")
        if len(safe_parity) != self.nsym * 2:
            raise ValueError(f"OCR-safe parity must be {self.nsym * 2} characters")

        safe_base = len(self.OCR_SAFE_ALPHABET)
        parity = []

        for i in range(0, len(safe_parity), 2):
            high_char = safe_parity[i]
            low_char = safe_parity[i + 1]

            if high_char not in self._ocr_safe_index or low_char not in self._ocr_safe_index:
                raise ValueError("OCR-safe parity contains an unsupported character")

            symbol = self._ocr_safe_index[high_char] * safe_base + self._ocr_safe_index[low_char]

            if symbol > self.FIELD_MAX:
                raise ValueError("OCR-safe parity contains an invalid symbol pair")

            parity.append(symbol)

        return parity

    def encode_with_ocr_safe_parity(self, message):
        """
        Encode message and return the parity as OCR-safe characters.

        Returns:
            message_symbols, safe_parity
        """
        codeword = self.encode(message)
        return codeword[:-self.nsym], self.parity_to_ocr_safe(codeword[-self.nsym:])

    def codeword_from_ocr_safe_parity(self, message, safe_parity):
        """
        Build a byte-oriented codeword from message symbols and OCR-safe parity.
        """
        message = self._validate_symbols(message, "message")
        if len(message) + self.nsym > self.FIELD_MAX:
            raise ValueError("Message is too long for GF(256). Maximum codeword length is 255 symbols.")

        return list(message) + self.ocr_safe_to_parity(safe_parity)

    def correct_with_ocr_safe_parity(self, message, safe_parity):
        """
        Correct a message using OCR-safe parity characters.
        """
        return self.correct(self.codeword_from_ocr_safe_parity(message, safe_parity))

    # ============================================================
    # 6. Syndrome calculation
    # ============================================================

    def calc_syndromes(self, codeword):
        """
        Calculate syndromes.

        If all syndromes are zero, there is no detectable error.
        """
        codeword = self._validate_codeword(codeword)
        syndromes = [0]

        for i in range(self.nsym):
            syndromes.append(self.poly_eval(codeword, self.gf_pow(2, i)))

        return syndromes

    def check(self, codeword):
        """
        Check whether codeword has no detectable errors.
        """
        syndromes = self.calc_syndromes(codeword)
        return max(syndromes) == 0

    # ============================================================
    # 7. Error locator polynomial
    # ============================================================

    def find_error_locator(self, syndromes):
        """
        Find error locator polynomial using Berlekamp-Massey algorithm.

        This polynomial helps identify the positions of corrupted symbols.
        """
        syndromes = self._validate_symbols(syndromes, "syndromes")
        if len(syndromes) < self.nsym + 1:
            raise ValueError(f"syndromes must contain at least {self.nsym + 1} symbols")

        err_loc = [1]
        old_loc = [1]

        for i in range(self.nsym):
            # Calculate discrepancy.
            delta = syndromes[i + 1]

            for j in range(1, len(err_loc)):
                delta ^= self.gf_mul(err_loc[-(j + 1)], syndromes[i + 1 - j])

            old_loc.append(0)

            if delta != 0:
                if len(old_loc) > len(err_loc):
                    new_loc = self.poly_scale(old_loc, delta)
                    old_loc = self.poly_scale(err_loc, self.gf_inverse(delta))
                    err_loc = new_loc

                err_loc = self.poly_add(err_loc, self.poly_scale(old_loc, delta))

        # Remove leading zeros.
        while len(err_loc) > 0 and err_loc[0] == 0:
            del err_loc[0]

        error_count = len(err_loc) - 1

        if error_count * 2 > self.nsym:
            raise ValueError("Too many errors to correct")

        return err_loc

    def find_errors(self, err_loc, codeword_length):
        """
        Find error positions using Chien search.

        Returns:
            list of error positions in the codeword
        """
        err_loc = self._validate_polynomial(err_loc, "err_loc")
        if not isinstance(codeword_length, int) or isinstance(codeword_length, bool):
            raise TypeError("codeword_length must be an integer")
        if codeword_length <= 0:
            raise ValueError("codeword_length must be positive")
        if codeword_length > self.FIELD_MAX:
            raise ValueError("codeword_length must not exceed 255")

        error_count = len(err_loc) - 1
        error_positions = []

        for i in range(codeword_length):
            x = self.gf_pow(2, i)

            if self.poly_eval(err_loc, x) == 0:
                error_positions.append(codeword_length - 1 - i)

        if len(error_positions) != error_count:
            raise ValueError("Could not locate all errors")

        return error_positions

    # ============================================================
    # 8. Error correction
    # ============================================================

    def find_errata_locator(self, coef_positions):
        """
        Compute errata locator polynomial from error coefficient positions.
        """
        coef_positions = self._validate_symbols(coef_positions, "coef_positions")
        errata_locator = [1]

        for pos in coef_positions:
            term = self.poly_add([1], [self.gf_pow(2, pos), 0])
            errata_locator = self.poly_mul(errata_locator, term)

        return errata_locator

    def find_error_evaluator(self, syndromes, err_loc, nsym):
        """
        Compute error evaluator polynomial.
        """
        syndromes = self._validate_polynomial(syndromes, "syndromes")
        err_loc = self._validate_polynomial(err_loc, "err_loc")
        if not isinstance(nsym, int) or isinstance(nsym, bool):
            raise TypeError("nsym must be an integer")
        if nsym <= 0:
            raise ValueError("nsym must be positive")

        _, remainder = self.poly_div(
            self.poly_mul(syndromes, err_loc),
            [1] + [0] * (nsym + 1),
        )

        return remainder

    def correct_errata(self, codeword, syndromes, error_positions):
        """
        Correct errors using Forney algorithm.
        """
        corrected = self._validate_codeword(codeword)
        syndromes = self._validate_symbols(syndromes, "syndromes")
        error_positions = self._validate_symbols(error_positions, "error_positions")

        if len(syndromes) < self.nsym + 1:
            raise ValueError(f"syndromes must contain at least {self.nsym + 1} symbols")
        if len(error_positions) != len(set(error_positions)):
            raise ValueError("error_positions must not contain duplicates")

        for position in error_positions:
            if position >= len(corrected):
                raise ValueError("error_positions must be within the codeword")

        # Convert positions to coefficient positions.
        coef_positions = [len(corrected) - 1 - pos for pos in error_positions]

        err_loc = self.find_errata_locator(coef_positions)

        err_eval = self.find_error_evaluator(
            syndromes[::-1],
            err_loc,
            len(err_loc) - 1,
        )[::-1]

        X = []

        for pos in coef_positions:
            # Equivalent to alpha^(-pos).
            X.append(self.gf_pow(2, -(self.FIELD_MAX - pos)))

        error_magnitudes = [0] * len(codeword)

        for i, Xi in enumerate(X):
            Xi_inv = self.gf_inverse(Xi)

            # Formal derivative part.
            err_loc_prime = 1

            for j, Xj in enumerate(X):
                if i != j:
                    err_loc_prime = self.gf_mul(
                        err_loc_prime,
                        self.gf_sub(1, self.gf_mul(Xi_inv, Xj)),
                    )

            y = self.poly_eval(err_eval[::-1], Xi_inv)
            y = self.gf_mul(Xi, y)
            magnitude = self.gf_div(y, err_loc_prime)

            error_magnitudes[error_positions[i]] = magnitude

        corrected = self.poly_add(corrected, error_magnitudes)

        return corrected

    def correct(self, codeword):
        """
        Correct a Reed-Solomon encoded codeword.

        Args:
            codeword: received message + parity

        Returns:
            original message without parity
        """
        received = self._validate_codeword(codeword)

        syndromes = self.calc_syndromes(received)

        # No error.
        if max(syndromes) == 0:
            return received[:-self.nsym]

        err_loc = self.find_error_locator(syndromes)

        # Reverse locator for Chien search.
        error_positions = self.find_errors(err_loc[::-1], len(received))

        corrected = self.correct_errata(received, syndromes, error_positions)

        # Verify correction.
        check_syndromes = self.calc_syndromes(corrected)

        if max(check_syndromes) > 0:
            raise ValueError("Could not correct message")

        return corrected[:-self.nsym]

    # ============================================================
    # 9. Helper functions
    # ============================================================

    @staticmethod
    def bytes_to_symbols(data):
        """
        Convert bytes to list of integers.
        """
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("data must be bytes or bytearray")

        return list(data)

    @staticmethod
    def symbols_to_bytes(symbols):
        """
        Convert list of integers to bytes.
        """
        return bytes(symbols)


# Compatibility wrappers for existing module-level callers.
def rs_encode_msg(message, nsym):
    return ReedSolomonForOcr(nsym).encode(message)


def rs_calc_syndromes(codeword, nsym):
    return ReedSolomonForOcr(nsym).calc_syndromes(codeword)


def rs_check(codeword, nsym):
    return ReedSolomonForOcr(nsym).check(codeword)


def rs_correct_msg(codeword, nsym):
    return ReedSolomonForOcr(nsym).correct(codeword)


def rs_encode_msg_with_ocr_safe_parity(message, nsym):
    return ReedSolomonForOcr(nsym).encode_with_ocr_safe_parity(message)


def rs_correct_msg_with_ocr_safe_parity(message, safe_parity, nsym):
    return ReedSolomonForOcr(nsym).correct_with_ocr_safe_parity(message, safe_parity)


def bytes_to_symbols(data):
    return ReedSolomonForOcr.bytes_to_symbols(data)


def symbols_to_bytes(symbols):
    return ReedSolomonForOcr.symbols_to_bytes(symbols)
