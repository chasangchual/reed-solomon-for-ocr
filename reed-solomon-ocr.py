"""
Educational Reed-Solomon implementation over GF(256).

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

    def __init__(self, nsym, primitive_poly=PRIMITIVE_POLY):
        if nsym <= 0:
            raise ValueError("nsym must be positive")
        if nsym >= self.FIELD_MAX:
            raise ValueError("nsym must be less than 255")

        self.nsym = nsym
        self.primitive_poly = primitive_poly
        self.gf_exp, self.gf_log = self._init_tables(primitive_poly)

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

        return gf_exp, gf_log

    # ============================================================
    # 2. GF(256) arithmetic
    # ============================================================

    @staticmethod
    def gf_add(x, y):
        """
        Addition in GF(256).

        In binary finite fields, addition is XOR.
        """
        return x ^ y

    @staticmethod
    def gf_sub(x, y):
        """
        Subtraction in GF(256).

        Same as addition because XOR is its own inverse.
        """
        return x ^ y

    def gf_mul(self, x, y):
        """
        Multiplication in GF(256).
        """
        if x == 0 or y == 0:
            return 0

        return self.gf_exp[self.gf_log[x] + self.gf_log[y]]

    def gf_div(self, x, y):
        """
        Division in GF(256).
        """
        if y == 0:
            raise ZeroDivisionError("Division by zero in GF(256)")

        if x == 0:
            return 0

        return self.gf_exp[(self.gf_log[x] + self.FIELD_MAX - self.gf_log[y]) % self.FIELD_MAX]

    def gf_pow(self, x, power):
        """
        Power operation in GF(256).
        """
        if x == 0:
            return 0

        return self.gf_exp[(self.gf_log[x] * power) % self.FIELD_MAX]

    def gf_inverse(self, x):
        """
        Multiplicative inverse in GF(256).
        """
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
        return [self.gf_mul(coef, scalar) for coef in poly]

    def poly_add(self, p, q):
        """
        Add two polynomials over GF(256).
        Addition is XOR coefficient-wise.
        """
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
        result = [0] * (len(p) + len(q) - 1)

        for j, q_coef in enumerate(q):
            for i, p_coef in enumerate(p):
                result[i + j] ^= self.gf_mul(p_coef, q_coef)

        return result

    def poly_eval(self, poly, x):
        """
        Evaluate polynomial at x using Horner's method.
        """
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

    # ============================================================
    # 6. Syndrome calculation
    # ============================================================

    def calc_syndromes(self, codeword):
        """
        Calculate syndromes.

        If all syndromes are zero, there is no detectable error.
        """
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
        errata_locator = [1]

        for pos in coef_positions:
            term = self.poly_add([1], [self.gf_pow(2, pos), 0])
            errata_locator = self.poly_mul(errata_locator, term)

        return errata_locator

    def find_error_evaluator(self, syndromes, err_loc, nsym):
        """
        Compute error evaluator polynomial.
        """
        _, remainder = self.poly_div(
            self.poly_mul(syndromes, err_loc),
            [1] + [0] * (nsym + 1),
        )

        return remainder

    def correct_errata(self, codeword, syndromes, error_positions):
        """
        Correct errors using Forney algorithm.
        """
        corrected = list(codeword)

        # Convert positions to coefficient positions.
        coef_positions = [len(codeword) - 1 - pos for pos in error_positions]

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
        if len(codeword) > self.FIELD_MAX:
            raise ValueError("Codeword is too long for GF(256)")

        received = list(codeword)

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


def bytes_to_symbols(data):
    return ReedSolomonForOcr.bytes_to_symbols(data)


def symbols_to_bytes(symbols):
    return ReedSolomonForOcr.symbols_to_bytes(symbols)

