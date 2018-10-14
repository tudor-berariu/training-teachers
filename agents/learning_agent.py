from typing import Dict, List, Tuple


class LearningAgent:

    def process(self, data, target) -> Tuple[List[float], Dict[str, float]]:
        raise NotImplementedError
