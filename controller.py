from typing import List

from tfl_manager import TFLMan


def remove_endl(line: str):
    return line[:-1] if line and '\n' == line[-1] else line


class Controller:
    def __init__(self, pls_file) -> None:
        with open(pls_file, 'r') as pls_file:
            pkl_path: str = pls_file.readline()[:-1]
            assert 'pkl' == pkl_path.split('.')[-1]

            lines: List[str] = pls_file.readlines()
            assert remove_endl(lines[0]).isdigit()

            self.__frame_list: List[str] = lines[1:]
            assert any('png' == frame.split('.')[-1] for frame in self.__frame_list)

        self.__tfl_manager = TFLMan(pkl_path)

    def start(self) -> None:
        for i, frame in enumerate(self.__frame_list):
            frame = remove_endl(frame)
            self.__tfl_manager.run(frame, i)
