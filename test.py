import enum

class InitializationType(enum.Enum):
    no_fixes = "no fixes"
    avoid_being_confidently_wrong = "avoid being confidently_wrong"
    squash_h = "squash h"


initialization_type=InitializationType.avoid_being_confidently_wrong


if initialization_type == InitializationType.no_fixes:
    print("no_fixes")
elif initialization_type == InitializationType.avoid_being_confidently_wrong:
    print("avoid_being_confidently_wrong")
elif initialization_type == InitializationType.squash_h:
    print("squash_h")
else:
    print("else")