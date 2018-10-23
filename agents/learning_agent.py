from typing import Dict, List, Tuple


class LearningAgent:

    def process(self, data, target,
                nrmlz) -> Tuple[List[float], Dict[str, float]]:
        raise NotImplementedError

    def save_state(self, _out_dir, _epoch, _data, _target) -> None:
        raise NotImplementedError

    def init_student(self, _student, _optimizer_args, _nsamples) -> None:
        raise NotImplementedError
