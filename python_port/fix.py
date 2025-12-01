import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import rich

def main():
    fpath: Path = Path("data/train.json")
    if len(sys.argv) > 1:
        fpath = Path(sys.argv[-1])
    with open(fpath, "r") as fp:
        raw_stories: List[Dict[str, Any]] = json.load(fp)
    num_changed = 0
    for i, story in enumerate(raw_stories):
        for j, sentence in enumerate(story["sentences"]):
            try:
                annotation = sentence["annotation"]
            except KeyError as e:
                print(f"failed at story: {i+1} | sentence: {j+1}")
                rich.print(sentence)
                raise e
            fixed_roles = {}
            roles: Dict[str, str] = annotation["roles"]
            anaphora: Dict[str, str] = annotation["anaphora"]
            for role, actor in roles.items():
                for a, b in anaphora.items():
                    if actor == b:
                        num_changed += 1
                        actor = a
                fixed_roles[role] = actor
            annotation["roles"] = fixed_roles
            del annotation["arguments"]
        if story.get("key"):
            del story["key"]
    with open(fpath.with_stem(f"{fpath.stem}_fixed"), "w") as fp:
        json.dump(raw_stories, fp)
    print(num_changed)

if __name__ == '__main__':
    main()