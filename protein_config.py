from enum import Enum


class Protein(Enum):
    UBIQUITIN = (
        "Ubiquitin",
        "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
    )
    PROTEIN_G = (
        "Protein G",
        "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE",
    )
    PROTEIN_A = (
        "Protein A",
        "MNAAQHDEAQQNAFYQVLNMPNLNADQRNGFIQSLKDDPSQSANVLGEAQKLNDSQAPK",
    )
    SH3_DOMAIN = (
        "SH3 Domain",
        "DETGKELVLALYDYQEKSPREVTMKKGDILTLLNSTNKDWWKVEVNDRQGFVPAAYVKKLD",
    )
    HOMEODOMAIN = (
        "Homeodomain",
        "RRRKRTAEREAELQKIVSEPGDSVKKKEGERLKQLYIEQSNKNRAIKRLEIQ",
    )
    ZINC_FINGER = ("Zinc Finger", "YKCGLCERSFVEKSALSRHQKRHTGEKPYK")
    WW_DOMAIN = ("WW Domain", "PLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPR")
    TRP_CAGE = ("Trp-Cage", "DAYAQWLKDGGPSSGRPPPS")

    @property
    def protein_name(self):
        return self.value[0]

    @property
    def sequence(self):
        return self.value[1]

    @classmethod
    def get_all_proteins(cls):
        """Returns list of (name, sequence) tuples"""
        return [(p.protein_name, p.sequence) for p in cls]
