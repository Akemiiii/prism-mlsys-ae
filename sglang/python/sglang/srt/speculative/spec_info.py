from enum import IntEnum, auto


class SpeculativeAlgorithm(IntEnum):
    NONE = auto()
    EAGLE = auto()
    EAGLE3 = auto()
    LD = auto()
    LD3 = auto()
    PRISM = auto()

    def is_none(self):
        return self == SpeculativeAlgorithm.NONE

    def is_eagle(self):
        return self in (SpeculativeAlgorithm.EAGLE, SpeculativeAlgorithm.EAGLE3, SpeculativeAlgorithm.LD3, SpeculativeAlgorithm.LD, SpeculativeAlgorithm.PRISM)
    
    def is_exact_eagle3(self):
        return self == SpeculativeAlgorithm.EAGLE3

    def is_eagle3(self):
        return self == SpeculativeAlgorithm.EAGLE3 or self == SpeculativeAlgorithm.LD3

    def is_ld3(self):
        return self == SpeculativeAlgorithm.LD3
    
    def is_ld(self):
        return self == SpeculativeAlgorithm.LD or self == SpeculativeAlgorithm.LD3 or self == SpeculativeAlgorithm.PRISM

    def is_prism(self):
        return self == SpeculativeAlgorithm.PRISM

    @staticmethod
    def from_string(name: str):
        name_map = {
            "EAGLE": SpeculativeAlgorithm.EAGLE,
            "EAGLE3": SpeculativeAlgorithm.EAGLE3,
            "LD": SpeculativeAlgorithm.LD,
            "LD3": SpeculativeAlgorithm.LD3,
            "PRISM": SpeculativeAlgorithm.PRISM,
            None: SpeculativeAlgorithm.NONE,
        }
        if name is not None:
            name = name.upper()
        return name_map[name]
