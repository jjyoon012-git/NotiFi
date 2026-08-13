"""Project-wide model and dataset constants."""

N_LINKS = 3
N_SUBCARRIERS = 114
N_IQ = 2
N_ACTIONS = 17
N_RISKS = 3

SOURCE_SITES = frozenset(
    {
        ("ajh", "E01"),
        ("ajh", "E02"),
        ("ajh", "E03"),
        ("mhw", "E01"),
        ("mhw", "E02"),
        ("mhw", "E03"),
        ("lmh", "E01"),
    }
)
SEALED_SITE = ("yja", "E02")
EXCLUDED_SITES = frozenset({("lmh", "E02"), ("lmh", "E03")})
