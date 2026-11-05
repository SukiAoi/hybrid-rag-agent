# -*- coding: utf-8 -*-
"""进程内限流器：按 IP 分钟级 + 日级，外加全站日配额。

单进程内存实现，重启即清零。对演示站点足够。
要多实例或持久化（重启不清零）就换成 Redis。
"""
import time
from threading import Lock


class RateLimiter:
    def __init__(self, per_min: int, per_day: int, global_day: int):
        self.per_min = per_min
        self.per_day = per_day
        self.global_day = global_day
        self._lock = Lock()
        self._ip_min: dict[str, tuple[float, int]] = {}   # ip -> (窗口起点, 次数)
        self._ip_day: dict[str, tuple[str, int]] = {}     # ip -> (日期, 次数)
        self._global: tuple[str, int] = (self._today(), 0)

    @staticmethod
    def _today() -> str:
        return time.strftime("%Y-%m-%d")

    def check(self, ip: str) -> tuple[bool, str]:
        """返回 (是否放行, 拒绝原因)。放行时已计入配额。"""
        now = time.time()
        today = self._today()

        with self._lock:
            g_day, g_cnt = self._global
            if g_day != today:
                g_day, g_cnt = today, 0

            # 1) 全站日配额：最硬的一道闸，防止被刷爆
            if g_cnt >= self.global_day:
                self._global = (g_day, g_cnt)
                return False, "今日站点总额度已用完，请明天再试。"

            # 2) 单 IP 日配额
            d, c = self._ip_day.get(ip, (today, 0))
            if d != today:
                d, c = today, 0
            if c >= self.per_day:
                self._ip_day[ip] = (d, c)
                self._global = (g_day, g_cnt)
                return False, f"单日提问上限 {self.per_day} 次，请明天再来。"

            # 3) 单 IP 分钟窗口
            w, n = self._ip_min.get(ip, (now, 0))
            if now - w >= 60:
                w, n = now, 0
            if n >= self.per_min:
                self._ip_min[ip] = (w, n)
                self._ip_day[ip] = (d, c)
                self._global = (g_day, g_cnt)
                wait = max(1, int(60 - (now - w)))
                return False, f"请求太频繁，请 {wait} 秒后再试。"

            # 通过，计数
            self._ip_min[ip] = (w, n + 1)
            self._ip_day[ip] = (d, c + 1)
            self._global = (g_day, g_cnt + 1)
            return True, ""

    def snapshot(self) -> dict:
        """用于 /stats 或排障"""
        with self._lock:
            return {
                "date": self._global[0],
                "global_used": self._global[1],
                "global_quota": self.global_day,
                "unique_ips": len(self._ip_day),
                "per_min": self.per_min,
                "per_day": self.per_day,
            }
