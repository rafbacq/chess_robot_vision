# Stockfish engine wrapper

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import chess
import chess.engine


@dataclass
class StockfishConfig:
    path: str
    time_limit_s: float = 0.2
    skill_level: Optional[int] = None


class StockfishEngine:
    def __init__(self, cfg: StockfishConfig):
        self.cfg = cfg
        self.engine = chess.engine.SimpleEngine.popen_uci(cfg.path)
        if cfg.skill_level is not None:
            try:
                self.engine.configure({"Skill Level": int(cfg.skill_level)})
            except Exception:
                pass

    def best_move(self, board: chess.Board) -> chess.Move:
        res = self.engine.play(board, chess.engine.Limit(time=self.cfg.time_limit_s))
        return res.move

    def close(self) -> None:
        self.engine.quit()