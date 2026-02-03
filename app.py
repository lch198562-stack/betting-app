import streamlit as st
import numpy as np
from scipy import stats
import re
import math
from datetime import datetime
from PIL import Image
import pytesseract
from typing import Dict, List, Tuple, Optional

# 設定頁面
st.set_page_config(page_title="智能投注系統 V6.9.2", page_icon="⚽", layout="wide")

import math
import re  
import random # 放在這裡
import numpy as np # 放在這裡
from scipy import stats
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

# ==========================================
# 🔒 [V6.0] 系統確定性鎖定 (System Lock)
# ==========================================
# 設定「隨機種子 (Seed)」，確保每次運算的隨機數都一樣
# 只要這個數字 (42) 不變，結果就永遠不會變
SEED_VALUE = 42

random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)

# 強制設定浮點數顯示精度，避免 0.00000001 的誤差
np.set_printoptions(precision=4, suppress=True)

print(f"🔒 系統已鎖定，種子碼: {SEED_VALUE} (確保每次計算結果一致)")
# ==========================================

# ═══════════════════════════════════════════════════════════════
# 【模組 1】🚀 V2.6.7 高階戰術運算模組 (波動/慢熱/疲勞)
# ═══════════════════════════════════════════════════════════════

class DataNormalizer:
@staticmethod
def _flip_score(score_str: str) -> str:
        """[內部工具] 將 'A-B' 反轉為 'B-A'"""
        if not score_str or '-' not in score_str: return score_str
        try:
            p1, p2 = score_str.split('-')
            return f"{p2}-{p1}"
        except: return score_str

    @staticmethod
    def _get_match_result(score_str: str, venue: str) -> str:
        """[內部工具] 計算標準賽果 (W/D/L)"""
        if not score_str or '-' not in score_str: return '?'
        try:
            h, a = map(int, score_str.split('-'))
            if h == a: return 'D'
            if venue == 'home': return 'W' if h > a else 'L'
            else: return 'W' if a > h else 'L' # 作客時：客隊分 > 主隊分 才算贏
        except: return '?'

    @staticmethod
    def _smart_fix_list(match_list: List[Dict], manual_form_str: str, team_name: str):
        """[核心邏輯] 增強版：加入 Debug 信息打印"""
        if not match_list or not manual_form_str: return
        manual_results = [c.upper() for c in manual_form_str if c.upper() in ['W', 'D', 'L']]
        
        print(f"\n🔍 [智能核對] 正在檢查 {team_name} 的數據邏輯...")
        
        for i, (match, user_result) in enumerate(zip(match_list, manual_results)):
            current_score = match.get('score', '')
            venue = match.get('venue', 'home')
            date = match.get('date', '未知日期')
            
            # 1. 系統嘗試解讀
            system_result = DataNormalizer._get_match_result(current_score, venue)
            
            # 2. 狀態顯示
            status = "✅ 吻合"
            action = "無須修改"
            
            if system_result != '?' and system_result != user_result:
                # 衝突發生，嘗試修復
                flipped_score = DataNormalizer._flip_score(current_score)
                flipped_result = DataNormalizer._get_match_result(flipped_score, venue)
                
                if flipped_result == user_result:
                    match['score'] = flipped_score
                    status = "🔧 已修復"
                    action = f"比分由 {current_score} 反轉為 {flipped_score}"
                    system_result = flipped_result # 更新結果
                else:
                    status = "⚠️ 無法修復"
                    action = f"數據({system_result})與手動({user_result})嚴重衝突"

            # 3. 打印診斷報告 (讓你一眼看清系統點樣諗)
            venue_str = "主場" if venue == 'home' else "作客"
            # 解讀邏輯顯示
            logic_str = ""
            if venue == 'home':
                h, a = map(int, match['score'].split('-'))
                logic_str = f"主({h}) vs 客({a})"
            else:
                h, a = map(int, match['score'].split('-'))
                logic_str = f"對手({h}) vs 我方({a})"

            print(f"   [{i+1}] {date} ({venue_str}) 比分:{match['score']:5} | 系統判定: {system_result} (邏輯:{logic_str}) | 用戶輸入: {user_result} | {status}")

    @staticmethod
    def normalize_relative_scores(match_data: Dict) -> Dict:
        """[主入口] 執行智能交叉驗證"""
        # 1. 修正主隊
        h_str = "".join(match_data.get('home_recent_form', []) if isinstance(match_data.get('home_recent_form'), list) else str(match_data.get('home_recent_form', '')))
        DataNormalizer._smart_fix_list(match_data.get('home_recent_matches_detailed', []), h_str, "主隊")

        # 2. 修正客隊
        a_str = "".join(match_data.get('away_recent_form', []) if isinstance(match_data.get('away_recent_form'), list) else str(match_data.get('away_recent_form', '')))
        DataNormalizer._smart_fix_list(match_data.get('away_recent_matches_detailed', []), a_str, "客隊")

        # 3. 修正 H2H
        h2h_str = "".join(match_data.get('h2h_recent_form', []) if isinstance(match_data.get('h2h_recent_form'), list) else str(match_data.get('h2h_recent_form', '')))
        DataNormalizer._smart_fix_list(match_data.get('h2h_details', []), h2h_str, "對賽往績")
        
        return match_data



class AdvancedMetrics:
    @staticmethod
    def _parse_date(date_str: str, current_date: datetime) -> Optional[datetime]:
        """[純Python] 智能解析日期，自動處理跨年問題"""
        if not date_str: return None
        formats = ["%y-%m-%d", "%Y-%m-%d", "%d/%m/%y", "%m-%d", "%d-%m"]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                # 如果格式只有月日 (e.g. "12-06")，需要推斷年份
                if "y" not in fmt and "Y" not in fmt:
                    dt = dt.replace(year=current_date.year)
                    # 如果日期比今天還未來 (例如現在1月，讀到12月)，說明是去年的
                    if dt > current_date + timedelta(days=30): 
                        dt = dt.replace(year=current_date.year - 1)
                return dt
            except: continue
        return None

    @staticmethod
    def sort_matches_by_date(matches_data: List[Dict], current_date: datetime = None) -> List[Dict]:
        """[純Python] 強制將比賽列表按日期由近到遠排序"""
        if not matches_data: return []
        if current_date is None: current_date = datetime.now()
        
        valid_matches = []
        for m in matches_data:
            d_str = m.get('date') or m.get('time') # 兼容不同欄位名
            dt = AdvancedMetrics._parse_date(d_str, current_date)
            if dt:
                m['_dt_obj'] = dt # 暫存 datetime 對象用於排序
                valid_matches.append(m)
        
        # 按日期降序排列 (最新的在 index 0)
        valid_matches.sort(key=lambda x: x['_dt_obj'], reverse=True)
        return valid_matches

    @staticmethod
    def calculate_volatility(scores_list: List[int]) -> float:
        """[純Python] 計算標準差 (替代 numpy.std)"""
        if not scores_list or len(scores_list) < 3: return 0.0
        try:
            n = len(scores_list)
            mean = sum(scores_list) / n
            variance = sum((x - mean) ** 2 for x in scores_list) / n
            return math.sqrt(variance)
        except: return 0.0

    @staticmethod
    def calculate_time_decay(matches_data: List[Dict]) -> Dict:
        total_goals = 0; second_half_goals = 0
        for m in matches_data:
            try:
                score_str = m.get('score', '0-0')
                ft_h, ft_a = map(int, score_str.split('-'))
                ft_total = ft_h + ft_a
                if 'ht_score' in m and '-' in m['ht_score']:
                    ht_h, ht_a = map(int, m['ht_score'].split('-'))
                    ht_total = ht_h + ht_a
                else: ht_total = ft_total / 2
                total_goals += ft_total; second_half_goals += (ft_total - ht_total)
            except: continue
        if total_goals == 0: return {'ratio': 0.5, 'label': '⚖️ 均衡型'}
        ratio = second_half_goals / total_goals
        if ratio >= 0.65: label = '🐢 慢熱型 (下半場發力)'
        elif ratio <= 0.35: label = '⚡ 快熱型 (上半場搶攻)'
        else: label = '⚖️ 均衡型'
        return {'ratio': ratio, 'label': label}

    @staticmethod
    def calculate_fatigue(matches_data: List[Dict], current_match_date: datetime) -> Dict:
        """[邏輯優化] 接收列表 -> 排序 -> 取最近一場計算"""
        # 這裡調用上方定義的 sort_matches_by_date
        sorted_matches = AdvancedMetrics.sort_matches_by_date(matches_data, current_match_date)
        
        if not sorted_matches: 
            return {'days': 7, 'status': '🟢 體力充沛 (無近期數據)'}
            
        last_match_date = sorted_matches[0]['_dt_obj']
        days_diff = (current_match_date - last_match_date).days
        
        if days_diff < 0: days_diff = 7 # 防錯
        
        if days_diff <= 3: status = '🔴 極度疲勞 (休息<=3天)'
        elif days_diff <= 5: status = '🟡 輕微疲勞 (休息4-5天)'
        else: status = '🟢 體力充沛'
        return {'days': days_diff, 'status': status}
        
    @staticmethod
    def calculate_weighted_momentum(form_list: List[str]) -> float:
        if not form_list: return 50.0
        score_map = {'W': 3, 'D': 1, 'L': 0}
        weights = [5, 4, 3, 2, 1]
        total_score = 0; total_weight = 0
        for i, result in enumerate(form_list[:5]):
            if i >= len(weights): break
            s = score_map.get(result.upper(), 1)
            w = weights[i]
            total_score += s * w
            total_weight += w
        if total_weight == 0: return 50.0
        max_possible_score = 3 * total_weight
        normalized = (total_score / max_possible_score) * 100
        return normalized


class HomeAwayDetailAnalyzer:
    @staticmethod
    def analyze_home_performance(matches, wins, draws, losses, goals_for, goals_against):
        if matches == 0: return {'win_rate': 0, 'home_advantage_score': 50}
        win_rate = wins / matches
        avg_gf = goals_for / matches
        score = (win_rate * 40) + (min(avg_gf/3, 1) * 30) + 30
        return {'win_rate': win_rate, 'home_advantage_score': score, 'avg_gf': avg_gf}

    @staticmethod
    def analyze_away_performance(matches, wins, draws, losses, goals_for, goals_against):
        if matches == 0: return {'win_rate': 0, 'away_strength_score': 50}
        win_rate = wins / matches
        avg_gf = goals_for / matches
        score = (win_rate * 50) + (min(avg_gf/2.5, 1) * 25) + 25
        return {'win_rate': win_rate, 'away_strength_score': score, 'avg_gf': avg_gf}

class HandicapHistoryAnalyzer:
    @staticmethod
    def analyze_handicap_performance(history: Dict, current_handicap: float) -> Dict:
        abs_h = abs(current_handicap)
        if abs_h <= 0.25: cat = 'flat'
        elif abs_h <= 0.75: cat = 'small'
        elif abs_h <= 1.5: cat = 'medium'
        else: cat = 'large'
        
        data = history.get(cat, {'matches': 0, 'covered': 0})
        matches = data.get('matches', 0)
        covered = data.get('covered', 0)
        
        rate = covered / matches if matches > 0 else 0.5
        adaptation_score = 50 + (rate - 0.5) * 40
        return {'category': cat, 'cover_rate': rate, 'adaptation_score': adaptation_score}

class DataValidator:
    @staticmethod
    def validate_match_data(match_data: Dict) -> Dict:
        required = ['home_team', 'away_team', 'league', 'handicap']
        for f in required:
            if f not in match_data: raise ValueError(f"❌ 缺少必填欄位：{f}")
        return match_data

class CompanyOddsManagerV229:
    COMPANIES = {'PIN': 'Pinnacle', 'B365': 'Bet365', 'CRO': 'Crown', '188': '188BET', 'HKJ': 'HKJC', 'WH': 'WilliamHill', 'INT': 'Interwetten'}
    PRIORITY_ORDER = ['PIN', 'B365', 'CRO', '188', 'HKJ', 'WH', 'INT']
def __init__(self, company_data: Dict):
        self.company_data = company_data
        self.available_companies = [c for c in self.PRIORITY_ORDER if c in company_data]
    def get_best_odds(self) -> Dict:
        if 'PIN' in self.company_data: return self._convert_odds(self.company_data['PIN'], 'PIN')
        if 'B365' in self.company_data: return self._convert_odds(self.company_data['B365'], 'B365')
        return self._calculate_average()
        def _convert_odds(self, odds_data: Dict, source: str) -> Dict:
        # 1. 提取原始數據
        curr_h = odds_data.get('current_home', 0)
        curr_a = odds_data.get('current_away', 0)
        early_h = odds_data.get('early_home')
        early_a = odds_data.get('early_away')
        
        # 用於記錄數據是否被修正 (符合 V3.0 完整性檢查)
        data_fixed_msg = None 

        # 2. 香港盤轉歐賠處理 (原有機制)
        if curr_h < 1.6: curr_h += 1.0 
        if curr_a < 1.6: curr_a += 1.0
        
        if early_h and early_h < 1.6: early_h += 1.0
        if early_a and early_a < 1.6: early_a += 1.0

        # 3. 🛡️ [V3.3 新增] 獨贏賠率防錯過濾 (Data Sanitization)
        # 架構原則：在數據層解決數據問題，不污染邏輯層
        odds_sum = curr_h + curr_a
        
        # 觸發條件：總和 > 4.5 (極大可能是獨贏) 或 單邊 > 3.0 (非正常亞盤)
        if odds_sum > 4.5 or curr_h > 3.0 or curr_a > 3.0:
            original_h, original_a = curr_h, curr_a
            # 強制重置為標準水位
            curr_h = 1.90
            curr_a = 1.90
            # 標記修正訊息，供報告顯示
            data_fixed_msg = f"⚠️ 異常賠率修正: {source} ({original_h:.2f}/{original_a:.2f}) -> 重置為 1.90"
            print(f"🚨 [系統攔截] {data_fixed_msg}")

        # 4. 計算變動 (原有機制)
        change = {'home_change': 0, 'trend_description': '平穩'}
        if early_h:
            chg = curr_h - early_h
            # 如果數據被修正過，變動值可能失真，這裡可以選擇歸零或保留
            if data_fixed_msg: chg = 0.0 
            
            desc = '主升' if chg > 0.02 else ('主跌' if chg < -0.02 else '平穩')
            change = {'home_change': chg, 'trend_description': desc}

        # 5. 回傳結果 (加入 data_fixed_msg)
        return {
            'home_odds': curr_h, 
            'away_odds': curr_a, 
            'source': source, 
            'source_name': self.COMPANIES.get(source, source), 
            'early_home': early_h, 
            'early_away': early_a, 
            'odds_change': change, 
            'all_companies_data': self._get_all_companies_comparison(),
            'debug_msg': data_fixed_msg  # ✅ 新增：將修正訊息傳遞出去
        }

        # ---------------------------------------------------------------------
        change = {'home_change': 0, 'trend_description': '平穩'}
        if early_h:
            chg = curr_h - early_h
            desc = '主升' if chg > 0.02 else ('    主跌'    if chg < -0.02 else '平穩')
            change = {'home_change': chg, 'trend_description': desc}
        return {'home_odds': curr_h, 'away_odds': curr_a, 'source': source, 'source_name': self.COMPANIES[source], 'early_home': early_h, 'early_away': early_a, 'odds_change': change, 'all_companies_data': self._get_all_companies_comparison()}
    def _calculate_average(self) -> Dict:
        return {'home_odds': 1.90, 'away_odds': 1.90, 'source': 'AVG', 'source_name': '平均', 'all_companies_data': {}}
    def _get_all_companies_comparison(self) -> Dict:
        comp = {}
        for c in self.available_companies:
            d = self.company_data[c]
            h = d.get('current_home', 0); a = d.get('current_away', 0)
            if h < 1.0: h += 1.0; 
            if a < 1.0: a += 1.0
            comp[c] = {'name': self.COMPANIES[c], 'home_odds': h, 'away_odds': a}
        return comp

class HandicapDictionary:
    CHINESE_ALIASES = {'平手': 0.0, '0': 0.0, '平': 0.0, '平半': 0.25, '0/0.5': 0.25, '0.25': 0.25, '半球': 0.5, '0.5': 0.5, '半': 0.5, '半一': 0.75, '0.5/1': 0.75, '0.75': 0.75, '一球': 1.0, '1': 1.0, '1.0': 1.0, '一球球半': 1.25, '1/1.5': 1.25, '1.25': 1.25, '球半': 1.5, '1.5': 1.5, '球半二球': 1.75, '1.5/2': 1.75, '1.75': 1.75, '二球': 2.0, '2': 2.0, '2.0': 2.0, '受讓平半': -0.25, '-0/0.5': -0.25, '-0.25': -0.25, '受讓半球': -0.5, '-0.5': -0.5, '受讓半一': -0.75, '-0.5/1': -0.75, '-0.75': -0.75, '受讓一球': -1.0, '-1': -1.0, '受讓一球球半': -1.25, '-1/1.5': -1.25, '-1.25': -1.25, '受讓球半': -1.5, '-1.5': -1.5}
    SORTED_ALIASES = sorted(CHINESE_ALIASES.items(), key=lambda x: len(x[0]), reverse=True)
    @staticmethod
    def identify_handicap(handicap_input) -> dict:
        text = str(handicap_input).strip().replace(' ', '')
        val = None
        for alias, value in HandicapDictionary.SORTED_ALIASES:
            if text == alias: val = value; break
        if val is None:
            try: val = float(text)
            except: val = 0.0
        display = f"{val}"
        if val > 0: display = f"主讓 {val}"
        elif val < 0: display = f"主受讓 {abs(val)}"
        else: display = "平手"
        return {'value': val, 'display': display}

class LeagueCharacteristicsAdapterV268:
    LEAGUE_CHARACTERISTICS = {
        'EPL': {'name': '英超', 'avg_goals': 2.8, 'draw_rate': 0.25, 'home_advantage': 0.60, 'home_bonus': 5.0, 'weight_adjustments': {'ranking_weight': 1.0, 'form_weight': 1.2, 'h2h_weight': 0.9}},
        'LaLiga': {'name': '西甲', 'avg_goals': 2.6, 'draw_rate': 0.28, 'home_advantage': 0.65, 'home_bonus': 6.0, 'weight_adjustments': {'ranking_weight': 1.1, 'form_weight': 1.0, 'h2h_weight': 1.1}},
        'Bundesliga': {'name': '德甲', 'avg_goals': 3.0, 'draw_rate': 0.23, 'home_advantage': 0.62, 'home_bonus': 5.0, 'weight_adjustments': {'ranking_weight': 0.9, 'form_weight': 1.1, 'h2h_weight': 0.8}},
        'SerieA': {'name': '意甲', 'avg_goals': 2.4, 'draw_rate': 0.30, 'home_advantage': 0.58, 'home_bonus': 5.0, 'weight_adjustments': {'ranking_weight': 1.2, 'form_weight': 0.9, 'h2h_weight': 1.2}},
        'Ligue1': {'name': '法甲', 'avg_goals': 2.7, 'draw_rate': 0.26, 'home_advantage': 0.61, 'home_bonus': 5.0, 'weight_adjustments': {'ranking_weight': 1.0, 'form_weight': 1.3, 'h2h_weight': 0.9}},
        'EFL_Championship': {'name': '英冠', 'avg_goals': 2.6, 'draw_rate': 0.27, 'home_advantage': 0.58, 'home_bonus': 6.0, 'weight_adjustments': {'ranking_weight': 0.9, 'form_weight': 1.3, 'h2h_weight': 1.0}},
        'EFL_League_One': {'name': '英甲', 'avg_goals': 2.5, 'draw_rate': 0.28, 'home_advantage': 0.56, 'home_bonus': 7.0, 'weight_adjustments': {'ranking_weight': 0.85, 'form_weight': 1.4, 'h2h_weight': 1.1}},
        'Segunda_Division': {'name': '西乙', 'avg_goals': 2.2, 'draw_rate': 0.31, 'home_advantage': 0.62, 'home_bonus': 6.0, 'weight_adjustments': {'ranking_weight': 1.0, 'form_weight': 1.1, 'h2h_weight': 1.0}},
        'Bundesliga_2': {'name': '德乙', 'avg_goals': 2.9, 'draw_rate': 0.25, 'home_advantage': 0.60, 'home_bonus': 6.0, 'weight_adjustments': {'ranking_weight': 0.85, 'form_weight': 1.2, 'h2h_weight': 0.9}},
        'SerieB': {'name': '意乙', 'avg_goals': 2.3, 'draw_rate': 0.32, 'home_advantage': 0.58, 'home_bonus': 6.0, 'weight_adjustments': {'ranking_weight': 1.1, 'form_weight': 1.0, 'h2h_weight': 1.0}},
        'Ligue2': {'name': '法乙', 'avg_goals': 2.5, 'draw_rate': 0.28, 'home_advantage': 0.59, 'home_bonus': 6.0, 'weight_adjustments': {'ranking_weight': 0.9, 'form_weight': 1.4, 'h2h_weight': 1.0}},
        'Eredivisie': {'name': '荷甲', 'avg_goals': 3.1, 'draw_rate': 0.24, 'home_advantage': 0.63, 'home_bonus': 5.0, 'weight_adjustments': {'ranking_weight': 1.0, 'form_weight': 1.1, 'h2h_weight': 0.8}},
        'Eerste_Divisie': {'name': '荷乙', 'avg_goals': 2.9, 'draw_rate': 0.26, 'home_advantage': 0.61, 'home_bonus': 6.0, 'weight_adjustments': {'ranking_weight': 0.85, 'form_weight': 1.3, 'h2h_weight': 0.9}},
        'Belgian_Pro_League': {'name': '比甲', 'avg_goals': 2.8, 'draw_rate': 0.25, 'home_advantage': 0.62, 'home_bonus': 5.0, 'weight_adjustments': {'ranking_weight': 0.95, 'form_weight': 1.2, 'h2h_weight': 0.9}},
        'Primeira_Liga': {'name': '葡超', 'avg_goals': 2.6, 'draw_rate': 0.27, 'home_advantage': 0.64, 'home_bonus': 6.0, 'weight_adjustments': {'ranking_weight': 1.15, 'form_weight': 1.0, 'h2h_weight': 1.0}},
        'Scottish_Premiership': {'name': '蘇超', 'avg_goals': 2.7, 'draw_rate': 0.26, 'home_advantage': 0.61, 'home_bonus': 6.0, 'weight_adjustments': {'ranking_weight': 1.1, 'form_weight': 1.1, 'h2h_weight': 1.0}},
        'Russian_Premier_League': {'name': '俄超', 'avg_goals': 2.5, 'draw_rate': 0.29, 'home_advantage': 0.66, 'home_bonus': 7.0, 'weight_adjustments': {'ranking_weight': 1.1, 'form_weight': 1.0, 'h2h_weight': 1.1}},
        'Allsvenskan': {'name': '瑞超', 'avg_goals': 2.6, 'draw_rate': 0.27, 'home_advantage': 0.60, 'home_bonus': 6.0, 'weight_adjustments': {'ranking_weight': 0.95, 'form_weight': 1.2, 'h2h_weight': 0.95}},
        'Eliteserien': {'name': '挪超', 'avg_goals': 2.8, 'draw_rate': 0.25, 'home_advantage': 0.62, 'home_bonus': 6.0, 'weight_adjustments': {'ranking_weight': 0.9, 'form_weight': 1.3, 'h2h_weight': 0.9}},
        'Brasileirao': {'name': '巴甲', 'avg_goals': 2.5, 'draw_rate': 0.28, 'home_advantage': 0.75, 'home_bonus': 10.0, 'weight_adjustments': {'ranking_weight': 0.95, 'form_weight': 1.3, 'h2h_weight': 1.1}},
        'Brasileirao_B': {'name': '巴乙', 'avg_goals': 2.1, 'draw_rate': 0.30, 'home_advantage': 0.68, 'home_bonus': 8.0, 'weight_adjustments': {'ranking_weight': 1.0, 'form_weight': 1.1, 'h2h_weight': 1.0}},
        'Argentine_Primera': {'name': '阿甲', 'avg_goals': 2.4, 'draw_rate': 0.30, 'home_advantage': 0.68, 'home_bonus': 8.0, 'weight_adjustments': {'ranking_weight': 1.0, 'form_weight': 1.2, 'h2h_weight': 1.2}},
        'Liga_MX': {'name': '墨超', 'avg_goals': 2.7, 'draw_rate': 0.26, 'home_advantage': 0.64, 'home_bonus': 6.0, 'weight_adjustments': {'ranking_weight': 0.95, 'form_weight': 1.2, 'h2h_weight': 1.0}},
        'MLS': {'name': '美職', 'avg_goals': 2.9, 'draw_rate': 0.24, 'home_advantage': 0.59, 'home_bonus': 5.0, 'weight_adjustments': {'ranking_weight': 0.85, 'form_weight': 1.4, 'h2h_weight': 0.8}},
        'Chilean_Primera': {'name': '智甲', 'avg_goals': 2.6, 'draw_rate': 0.27, 'home_advantage': 0.65, 'home_bonus': 7.0, 'weight_adjustments': {'ranking_weight': 0.9, 'form_weight': 1.3, 'h2h_weight': 1.1}},
        'J1_League': {'name': '日職', 'avg_goals': 2.6, 'draw_rate': 0.27, 'home_advantage': 0.58, 'home_bonus': 5.0, 'weight_adjustments': {'ranking_weight': 1.0, 'form_weight': 1.2, 'h2h_weight': 1.0}},
        'J2_League': {'name': '日乙', 'avg_goals': 2.5, 'draw_rate': 0.28, 'home_advantage': 0.56, 'home_bonus': 6.0, 'weight_adjustments': {'ranking_weight': 0.9, 'form_weight': 1.3, 'h2h_weight': 1.0}},
        'K_League': {'name': '韓職', 'avg_goals': 2.7, 'draw_rate': 0.26, 'home_advantage': 0.60, 'home_bonus': 6.0, 'weight_adjustments': {'ranking_weight': 0.95, 'form_weight': 1.2, 'h2h_weight': 0.95}},
        'A_League': {'name': '澳職', 'avg_goals': 2.8, 'draw_rate': 0.25, 'home_advantage': 0.57, 'home_bonus': 5.0, 'weight_adjustments': {'ranking_weight': 0.85, 'form_weight': 1.3, 'h2h_weight': 0.9}},
        'Saudi_Pro_League': {'name': '沙特聯', 'avg_goals': 2.6, 'draw_rate': 0.27, 'home_advantage': 0.65, 'home_bonus': 7.0, 'weight_adjustments': {'ranking_weight': 1.0, 'form_weight': 1.1, 'h2h_weight': 1.0}},
        'Super_Lig': {'name': '土超', 'avg_goals': 2.8, 'draw_rate': 0.26, 'home_advantage': 0.68, 'home_bonus': 8.0, 'weight_adjustments': {'ranking_weight': 1.0, 'form_weight': 1.2, 'h2h_weight': 1.1}},
        'Swiss_Super_League': {'name': '瑞士超', 'avg_goals': 3.1, 'draw_rate': 0.24, 'home_advantage': 0.58, 'home_bonus': 5.0, 'weight_adjustments': {'ranking_weight': 0.9, 'form_weight': 1.3, 'h2h_weight': 0.9}},
        'Austrian_Bundesliga': {'name': '奧甲', 'avg_goals': 3.0, 'draw_rate': 0.25, 'home_advantage': 0.60, 'home_bonus': 6.0, 'weight_adjustments': {'ranking_weight': 1.1, 'form_weight': 1.1, 'h2h_weight': 0.9}},
        'UAE_Pro_League': {'name': '阿聯酋超', 'avg_goals': 3.2, 'draw_rate': 0.22, 'home_advantage': 0.57, 'home_bonus': 5.0, 'weight_adjustments': {'ranking_weight': 1.0, 'form_weight': 1.3, 'h2h_weight': 0.9}},
        'Qatar_Stars_League': {'name': '卡塔爾聯', 'avg_goals': 3.1, 'draw_rate': 0.21, 'home_advantage': 0.58, 'home_bonus': 5.0, 'weight_adjustments': {'ranking_weight': 1.0, 'form_weight': 1.3, 'h2h_weight': 0.9}},
        'UEFA_Champions_League': {'name': '歐聯', 'avg_goals': 2.9, 'draw_rate': 0.24, 'home_advantage': 0.58, 'home_bonus': 6.0, 'weight_adjustments': {'ranking_weight': 1.3, 'form_weight': 1.0, 'h2h_weight': 1.2}},
        'UEFA_Europa_League': {'name': '歐霸', 'avg_goals': 2.7, 'draw_rate': 0.26, 'home_advantage': 0.60, 'home_bonus': 5.0, 'weight_adjustments': {'ranking_weight': 1.1, 'form_weight': 1.2, 'h2h_weight': 1.0}},
        'DEFAULT': {'name': '通用聯賽', 'avg_goals': 2.5, 'draw_rate': 0.27, 'home_advantage': 0.60, 'home_bonus': 5.0, 'weight_adjustments': {'ranking_weight': 1.0, 'form_weight': 1.0, 'h2h_weight': 1.0}},
    }
    @staticmethod
    def get_league_adjustments(league: str) -> Dict:
        return LeagueCharacteristicsAdapterV268.LEAGUE_CHARACTERISTICS.get(league, LeagueCharacteristicsAdapterV268.LEAGUE_CHARACTERISTICS['DEFAULT']).copy()

class SmartNoRecommendationSystem:
    @staticmethod
    def evaluate_recommendation_quality(adjusted_kelly, combined_draw_risk, defense_level, recommended_side):
        reasons = []; should_recommend = True; recommendation_type = '標準推薦'; confidence_level = '中'
        if adjusted_kelly < 0.03: should_recommend = False; reasons.append(f"❌ 凱利值過低 ({adjusted_kelly:.2%})")
        if combined_draw_risk > 0.35: should_recommend = False; reasons.append(f"❌ 平局風險過高 ({combined_draw_risk:.1%})")
        return {'should_recommend': should_recommend, 'recommendation_type': recommendation_type, 'confidence_level': confidence_level, 'comprehensive_score': 75, 'reasons': reasons, 'alternative': None}

class KellyOptimizer227:
 @staticmethod
def calculate_optimal_bet(kelly_value, bankroll, risk_preference, confidence_level):
        bet = bankroll * kelly_value * 0.5
        return {'adjusted_kelly_bet': bet, 'bet_percentage': bet/bankroll, 'risk_level': '低'}

class H2HDeepAnalyzer:
    """
    [V6.3 Module Update] H2H 深度分析器 (含時間衰減 Time-Decay)
    符合原則：模組化新增、雙向鏡像、系統性修復
    """
    
    @staticmethod
 def _parse_h2h_date(date_str: str) -> Optional[datetime]:
        """[內部工具] 智能解析 H2H 日期 (自動補全年份，防止 1900 年誤判)"""
        if not date_str: return None
        now = datetime.now()
        current_year = now.year
        
        # 1. 嘗試完整格式 (含年份)
        formats_with_year = ["%Y-%m-%d", "%y-%m-%d", "%d/%m/%y", "%d/%m/%Y", "%Y/%m/%d"]
        for fmt in formats_with_year:
            try: return datetime.strptime(date_str, fmt)
            except: continue
            
        # 2. 嘗試簡寫格式 (無年份，例如 05-12 或 12/05) -> 自動補全邏輯
        formats_no_year = ["%m-%d", "%d-%m", "%m/%d", "%d/%m"]
        for fmt in formats_no_year:
            try:
                dt = datetime.strptime(date_str, fmt)
                # 預設是 1900，強制替換為今年
                try:
                    dt = dt.replace(year=current_year)
                except ValueError: 
                    # 處理閏年 2月29日 的特殊情況 (如果今年不是閏年)
                    dt = dt.replace(year=current_year, day=28)

                # 邏輯修正：如果補完年份後發現日期在「未來」，證明是去年的比賽
                # 例如：現在是 1月，讀到 "12-25"，應該是去年的聖誕節
                if dt > now:
                    dt = dt.replace(year=current_year - 1)
                return dt
            except: continue
            
        return None

 @staticmethod
def _get_time_weight(match_date_str: str) -> float:
        """
        [核心數學] 計算時間權重 (Time Decay)
        原則：通用性 (General Applicability)
        """
        if not match_date_str: return 1.0 # 無日期則默認全權重 (兼容舊數據)
        
        match_dt = H2HDeepAnalyzer._parse_h2h_date(match_date_str)
        if not match_dt: return 1.0 
        
        # 計算距離現在的天數
        days_diff = (datetime.now() - match_dt).days
        
        # ⚖️ 時間衰減公式 (V6.3 System Logic)
        if days_diff < 180: return 1.2   # 半年內：加權 120% (近期戰績最具參考性)
        elif days_diff < 365: return 1.0 # 一年內：標準 100%
        elif days_diff < 730: return 0.8 # 兩年內：打 8 折
        elif days_diff < 1095: return 0.6 # 三年內：打 6 折
        else: return 0.3                 # 三年以上：只剩 30% (歷史塵埃，防止誤導)

  @staticmethod
def analyze_h2h_handicap(h2h_matches: List[Dict], current_handicap: float) -> Dict:
        """
        [主邏輯] 計算加權後的剋星關係
        原則：雙向鏡像 (Dual-Mirror Principle)
        """
        if not h2h_matches: 
            return {'nemesis_relationship': '無歷史數據', 'similar_handicap_cover_rate': 0.5}
        
        total_weight = 0.0
        weighted_my_wins = 0.0
        
        similar_hdp_wins = 0
        similar_count = 0
        
        # Debug Log (方便核對)
        print(f"\n⚔️ [H2H 時間加權分析] (當前盤口: {current_handicap})")
        
        for m in h2h_matches:
            try:
                score_str = m.get('score', '')
                if '-' not in score_str: continue
                
                # 1. 數據提取
                home_score, away_score = map(int, score_str.split('-'))
                venue = m.get('venue', 'home') 
                date_str = m.get('date', '')
                
                # 2. 計算權重
                weight = H2HDeepAnalyzer._get_time_weight(date_str)
                
                # 3. 勝負判斷 (⚖️ 雙向鏡像：主客邏輯完全對稱)
                is_my_win = False
                my_margin = 0
                
                if venue == 'home':
                    # 我方是主隊
                    if home_score > away_score: is_my_win = True
                    my_margin = home_score - away_score
                else:
                    # 我方是客隊
                    if away_score > home_score: is_my_win = True
                    my_margin = away_score - home_score
                
                # 4. 累加邏輯
                total_weight += weight
                if is_my_win: 
                    weighted_my_wins += weight
                    
                # 5. 相似盤口 (保留舊邏輯)
                hist_hdp = m.get('handicap', 0)
                if abs(hist_hdp - current_handicap) <= 0.25:
                    similar_count += 1
                    if my_margin > hist_hdp: similar_hdp_wins += 1
                
                # Debug Print
                # print(f"   - {date_str} ({venue}): {score_str} | 權重 {weight:.1f} | {'贏' if is_my_win else '輸'}")

            except Exception:
                continue

        if total_weight == 0: return {'nemesis_relationship': '無有效數據', 'similar_handicap_cover_rate': 0.5}

        # 6. 計算「加權勝率」
        weighted_win_rate = weighted_my_wins / total_weight
        
        # 7. 判定剋星關係 (輸出格式保持兼容，供 FinalJudge 讀取)
        nemesis = '互有勝負'
        
        # 門檻邏輯 (Thresholds)
        if weighted_win_rate >= 0.65: 
            nemesis = f'主隊明顯剋星 (加權勝率{weighted_win_rate:.0%})' 
        elif weighted_win_rate <= 0.25:
            nemesis = f'客隊明顯剋星 (加權勝率{1-weighted_win_rate:.0%})'
        
        sim_rate = similar_hdp_wins / similar_count if similar_count > 0 else 0.5
        
        return {'nemesis_relationship': nemesis, 'similar_handicap_cover_rate': sim_rate}
        
    @staticmethod
   def simulate_handicap_history(h2h_matches: List[Dict], current_handicap: float) -> Dict:
        """
        [V5.1 功能保留] H2H 盤路回測引擎
        原則：完整性連動 (Integrity Check) - 必須保留此函數供 V5.1 邏輯調用
        """
        if not h2h_matches: 
            return {'backtest_win_rate': 0.5, 'msg': '無歷史對賽'}

        wins = 0; pushes = 0; total = 0
        
        for m in h2h_matches:
            try:
                score_str = m.get('score', '')
                if '-' not in score_str: continue
                
                h_score, a_score = map(int, score_str.split('-'))
                
                # 模擬結算
                adjusted_h_score = h_score + current_handicap
                
                total += 1
                if adjusted_h_score > a_score: wins += 1
                elif adjusted_h_score == a_score: pushes += 1
            except: continue
            
        if total == 0: return {'backtest_win_rate': 0.5, 'msg': '無有效比分'}
        
        win_rate = (wins + pushes) / total
        msg = f"歷史盤口回測(主{current_handicap:+.1f}): 近{total}場 贏{wins} 走{pushes} ({win_rate:.0%}不敗)"
        
        return {'backtest_win_rate': win_rate, 'total': total, 'msg': msg}

        
 @staticmethod
def simulate_handicap_history(h2h_matches: List[Dict], current_handicap: float) -> Dict:
        """
        [V5.1 新增] H2H 盤路回測引擎
        功能：將「當前盤口」代入「歷史比分」，計算該盤口的歷史勝率。
        """
        if not h2h_matches: 
            return {'backtest_win_rate': 0.5, 'msg': '無歷史對賽'}

        # 定義：current_handicap 是主隊視角 (例如 +2.0 代表主受讓2球)
        # 我們要計算的是「下注當前盤口 (主隊)」在歷史上的勝率
        wins = 0
        pushes = 0
        total = 0
        
        for m in h2h_matches:
            try:
                score_str = m.get('score', '')
                if '-' not in score_str: continue
                
                # 解析比分 (格式永遠是 左-右)
                # 關鍵：必須知道歷史場次中，誰是主隊
                # 這裡簡化邏輯：假設 score 格式已經被 DataNormalizer 修正為 "主-客"
                h_score, a_score = map(int, score_str.split('-'))
                
                # 模擬結算：主隊得分 + 盤口 vs 客隊得分
                # 例如：比分 2-3，盤口 +2.0 -> (2+2) vs 3 -> 4 vs 3 -> 主贏盤
                adjusted_h_score = h_score + current_handicap
                
                total += 1
                if adjusted_h_score > a_score:
                    wins += 1
                elif adjusted_h_score == a_score:
                    pushes += 1
            except: continue
            
        if total == 0: return {'backtest_win_rate': 0.5, 'msg': '無有效比分'}
        
        # 計算「主隊盤口」的不敗率 (贏+走)
        win_rate = (wins + pushes) / total
        
        msg = f"歷史盤口回測(主{current_handicap:+.1f}): 近{total}場 贏{wins} 走{pushes} ({win_rate:.0%}不敗)"
        
        return {'backtest_win_rate': win_rate, 'total': total, 'msg': msg}


        total = len(h2h_matches); home_wins = 0; similar_hdp_wins = 0; similar_count = 0
        
        for m in h2h_matches:
            try:
                h, a = map(int, m['score'].split('-'))
                if h > a: home_wins += 1
                
                # 檢查相似盤口 (誤差 0.25 內)
                hist_hdp = m.get('handicap', 0)
                if abs(hist_hdp - current_handicap) <= 0.25:
                    similar_count += 1
                    if h - a > hist_hdp: similar_hdp_wins += 1
            except: continue

        win_rate = home_wins / total
        if win_rate >= 0.7: nemesis = '主隊明顯剋星'
        elif win_rate <= 0.3: nemesis = '客隊明顯剋星'
        else: nemesis = '互有勝負'
        
        sim_rate = similar_hdp_wins / similar_count if similar_count > 0 else 0.5
        return {'nemesis_relationship': nemesis, 'similar_handicap_cover_rate': sim_rate}

class MultiDimensionalRiskEvaluator:
    @staticmethod
    def evaluate_comprehensive_risk(draw_risk, heavy_defeat_risk, defense_level, consistency, adaptation_score):
        draw_score = draw_risk * 100
        defeat_score = heavy_defeat_risk * 100
        
        def_map = {'🟢 正常': 10, '🟢 季後賽豁免': 10, '🟡 輕微崩潰': 40, '🟠 嚴重崩潰': 70, '🔴 防守崩潰': 90}
        def_score = def_map.get(defense_level, 50)
        
        cons_map = {'高度一致': 10, '基本一致': 30, '輕微分歧': 60, '嚴重分歧': 90}
        cons_score = cons_map.get(consistency, 50)
        
        adapt_risk = 100 - adaptation_score
        
        total_risk = (draw_score * 0.2) + (defeat_score * 0.2) + (def_score * 0.25) + (cons_score * 0.2) + (adapt_risk * 0.15)
        
        if total_risk >= 70: level = '🔴 極高風險'
        elif total_risk >= 50: level = '🟠 高風險'
        elif total_risk >= 30: level = '🟡 中風險'
        else: level = '🟢 低風險'
        
        return {'score': total_risk, 'level': level, 'details': f"防守{def_score}|平局{draw_score:.0f}|一致性{cons_score}"}
        
class LineupImpactAnalyzer:
    """[V3.6.4 Fix] 模組 2: 傷停影響分析器 (累計疊加版)"""
    @staticmethod
    def analyze_injury_impact(text: str, team_name: str) -> Tuple[float, str]:
        if not text: return 0.0, ""
        penalty = 0.0
        details = []
        
        # 定義關鍵字與權重
        keywords = {
            '被徵召': 15.0, '國家隊': 15.0, '十字韌帶': 12.0, '骨折': 10.0, 
            '手術': 10.0, '頭號射手': 12.0, '核心': 10.0, '隊長': 8.0, 
            '主力': 6.0, '停賽': 5.0, '紅牌': 5.0, '軟骨': 8.0, '撕裂': 5.0
        }
        
        # [核心修復] 使用 count() 計算出現次數，實現傷害疊加
        for kw, score in keywords.items():
            count = text.count(kw)
            if count > 0:
                # 特殊處理：避免 "十字韌帶撕裂" 同時觸發 "十字韌帶" 和 "撕裂" 導致過重
                # 這裡簡單處理：如果關鍵字是複合的，通常權重已包含
                total_score = score * count
                penalty += total_score
                details.append(f"{kw}x{count}")
                
        # 國家隊重複扣分修正
        if '被徵召' in text and '國家隊' in text: penalty -= 15.0
        
        # 設定上限，避免分數無限膨脹 (但提高上限至 60 以容納災難級)
        penalty = min(60.0, penalty)
        
        msg = ""
        if penalty > 0:
            msg = f"🚑 [{team_name}傷停] 觸發關鍵字{'、'.join(details)}，戰力修正-{penalty:.1f}"
            
        return penalty, msg


class DealerPsychologyEngine:
    """[V2.9.7] 模組 1: 莊家意圖 (Trap Gap)"""
    @staticmethod
    def detect_trap_gap(win_prob: float, actual_odds: float) -> Tuple[float, str]:
        if win_prob <= 0.1 or actual_odds <= 1.0: return 0.0, ""
        fair_odds = 1 / win_prob
        gap = actual_odds - fair_odds
        penalty = 0.0
        msg = ""
        if gap > 0.45:
            penalty = -30.0
            msg = f"🚨 [陷阱警報] 賠率({actual_odds:.2f})遠高於理論({fair_odds:.2f})，Gap+{gap:.2f}，極度異常！"
        elif gap > 0.25:
            penalty = -15.0
            msg = f"⚠️ [疑似誘盤] 賠率虛高(Gap+{gap:.2f})，存在隱患"
        elif gap < -0.15:
            penalty = 12.0
            msg = f"💎 [真實防範] 莊家壓低賠率(Gap{gap:.2f})，真實看好"
        return penalty, msg

class SpecificHandicapTrendAnalyzer:
    """[V2.9.7] 模組 3: 特定盤口歷史一致性 (補漏)"""
    @staticmethod
 def analyze_trend(history: List[Dict], current_hdp: float, team_type: str) -> Tuple[float, str]:
        if not history: return 0.0, ""
        
        # 1. 篩選：只看「同主客」且「同盤口(誤差0.05內)」的比賽
        target_matches = []
        for m in history:
            try:
                # 歷史盤口 (需確保數據源有此欄位，若無則跳過)
                hist_hdp = float(m.get('handicap', -999))
                # 盤口匹配 (例如現在讓0.25，歷史也要是0.25)
                if abs(hist_hdp - current_hdp) < 0.05:
                    target_matches.append(m)
            except: continue
            
        if not target_matches: return 0.0, "" # 無參考樣本
        
        # 2. 計算該盤口下的贏盤率
        win_cover = 0
        total = len(target_matches)
        
        for m in target_matches:
            # 簡單判斷贏盤 (這裡假設 score 是 '主-客')
            try:
                score = m.get('score', '0-0')
                h, a = map(int, score.split('-'))
                diff = h - a
                # 若是主隊讓球
                if current_hdp > 0:
                    if diff > current_hdp: win_cover += 1
                # 若是主隊受讓 (盤口為負)
                elif current_hdp < 0:
                    if diff > current_hdp: win_cover += 1 # 例如 -0.5，輸1球(-1) < -0.5 (輸盤)
            except: pass
            
        win_rate = win_cover / total
        
        # 3. 評分邏輯
        penalty = 0.0
        msg = ""
        
        if total >= 3: # 樣本數至少3場才具參考價值
            if win_rate <= 0.2: # 該盤口下極少贏盤
                penalty = -15.0
                msg = f"📉 [盤路魔咒] {team_type}在盤口({current_hdp})下近{total}場僅贏盤{win_rate:.0%}，極不適應"
            elif win_rate >= 0.8: # 該盤口下經常贏盤
                penalty = 10.0
                msg = f"🔥 [盤路強勢] {team_type}在盤口({current_hdp})下近{total}場贏盤{win_rate:.0%}，特別擅長"
                
        return penalty, msg

        
class HandicapRuleGenerator:
    @staticmethod
   def get_payout_rules(handicap_val: float, rec_side: str) -> str:
    """
    [V2.9.4 Patch] 修正客隊讓球時的文案邏輯 (雙向鏡像修復)
    縮排層級：4空格 (屬於 class 內部方法)
    """
    # 1. 計算推薦方的「有效盤口」 (Effective Handicap)
    # 邏輯定義：
    # 系統原始盤口 (handicap_val): 正數(+)代表主讓，負數(-)代表主受讓
    # 有效盤口 (eff_hdp): 轉換為「推薦隊伍」視角的盤口值
    
    eff_hdp = 0.0
    if rec_side == 'home':
        eff_hdp = handicap_val 
    else: 
        # ⚖️ 雙向鏡像原則：若推客隊，需將盤口符號反轉
        # 例：主受讓(-0.75) -> 客隊即為讓球(+0.75)
        eff_hdp = -handicap_val
        
    # 2. 判定屬性 (Giving vs Receiving)
    # 基於 HandicapDictionary 定義：正數為讓球(Giving)，負數為受讓(Receiving)
    is_giving = eff_hdp > 0
    
    abs_eff = abs(eff_hdp)
    base = int(abs_eff)
    fraction = abs_eff - base
    
    # 3. 根據小數部分生成規則 (通用邏輯)
    
    # A. 整數盤 (0, 1.0, 2.0...)
    if fraction == 0.0:
        if abs_eff == 0: return "平手盤：贏球全贏，打和走盤"
        if is_giving:
            return f"贏 {base+1} 球或以上全贏，剛好贏 {base} 球走盤"
        else:
            return f"輸 {base-1} 球或不輸全贏，剛好輸 {base} 球走盤"
            
    # B. 半球盤 (0.5, 1.5...)
    elif abs(fraction - 0.5) < 0.01:
        if is_giving:
            return f"贏 {base+1} 球或以上全贏，否則全輸"
        else:
            return f"輸 {base} 球或不輸全贏，輸 {base+1} 球全輸"
            
    # C. 平半盤 (0.25, 1.25...)
    elif abs(fraction - 0.25) < 0.01: 
        if is_giving: # 讓平半(-0.25)
            if base == 0: return "贏球全贏，打和輸半"
            return f"贏 {base+1} 球全贏，剛好贏 {base} 球輸半"
        else: # 受讓平半(+0.25)
            if base == 0: return "贏球全贏，打和贏半"
            return f"輸 {base} 球或不輸全贏，剛好輸 {base} 球贏半" 
            
    # D. 半一盤 (0.75, 1.75...)
    elif abs(fraction - 0.75) < 0.01: 
        if is_giving: # 讓半一(-0.75)
            return f"贏 {base+2} 球全贏，剛好贏 {base+1} 球贏半"
        else: # 受讓半一(+0.75)
            return f"輸 {base} 球或不輸全贏，剛好輸 {base+1} 球輸半"
    
    return f"規則計算中 (盤口:{eff_hdp:.2f})"
    

class MarketResonanceV6:
    """
    [V6.0 Final] 市場共振分析器 (1x2 & O/U & Kelly)
    """
    @staticmethod
    def get_theoretical_handicap(euro_odds: float) -> float:
        """歐賠轉亞盤對照表"""
        if euro_odds <= 1.0: return 0.0
        if euro_odds < 1.20: return 2.25
        if euro_odds < 1.30: return 1.75
        if euro_odds < 1.42: return 1.50
        if euro_odds < 1.55: return 1.25
        if euro_odds < 1.70: return 1.00
        if euro_odds < 1.90: return 0.75
        if euro_odds < 2.15: return 0.50
        if euro_odds < 2.45: return 0.25
        if euro_odds < 2.90: return 0.00
        return -0.25

    @staticmethod
    def analyze_market_forces(match_data: dict, current_handicap: float) -> dict:
        # 1. 提取 1x2
        euro_home = match_data.get('manual_1x2', {}).get('home', 0)
        
        # 2. 提取大小球 (包含初盤同即時)
        ou_data = match_data.get('manual_ou', {})
        ou_trend = ou_data.get('trend', 'Flat')
        
        # 3. 提取凱利
        kelly_data = match_data.get('manual_kelly', {})
        k_early = kelly_data.get('early', 0)
        k_curr = kelly_data.get('current', 0)
        
        if euro_home == 0: 
            return {'theo_diff': 0, 'ou_support': 'Neutral', 'kelly_signal': 'None', 'msg': '無數據'}

        # 計算理論盤口差異
        theo_hdp = MarketResonanceV6.get_theoretical_handicap(euro_home)
        diff = theo_hdp - current_handicap
        
        # 大小球共振邏輯 (利用 DataInjector 算好的 Trend)
        support = 'Neutral'
        if current_handicap > 0: # 主讓
            if ou_trend == 'OverDrop': support = 'Home'
            elif ou_trend == 'UnderDrop': support = 'Away'
        elif current_handicap < 0: # 客讓
            if ou_trend == 'OverDrop': support = 'Away'
            elif ou_trend == 'UnderDrop': support = 'Home'
            
        # 凱利風控偵測 (Kelly Guard)
        kelly_signal = 'Neutral'
        kelly_diff = 0
        if k_early > 0 and k_curr > 0:
            kelly_diff = k_curr - k_early
            
            # 情況 A: 莊家防範 (低水 + 下跌)
            if k_curr < 0.92 and kelly_diff <= -0.02:
                kelly_signal = 'Guard' # 真盤
            
            # 情況 B: 莊家誘盤 (高水 + 上升)
            elif k_curr > 0.96 and kelly_diff >= 0.02:
                kelly_signal = 'Trap' # 假盤
                
            # 情況 C: 極度防範 (極低水 < 0.88)
            elif k_curr < 0.88:
                kelly_signal = 'SuperGuard'

        return {
            'theo_hdp': theo_hdp,
            'theo_diff': diff,
            'ou_support': support,
            'euro_odds': euro_home,
            'ou_trend': ou_trend,
            'kelly_signal': kelly_signal,
            'kelly_diff': kelly_diff,
            'kelly_curr': k_curr
        }


class FinalJudgeV37_Clean:
    """
    [V6.9.1 IronLogic - Final Complete] 首席法官邏輯 (鐵血修正完整版)
    
    【架構層級 - 雙向鏡像確認 ✅】
    1. V6.9.1 新增：⛔ 進攻死線 (Offensive Dead-Line) - 針對無鋒陣(場均入球<0.6)重罰
    2. V6.9 新增：🛡️ 防火牆協議 (主客防守漏水均折抵)
    3. V6.9 新增：🎭 傷情虛實校驗 (主客深傷淺盤均回補，但啞火者除外)
    4. V6.8 保留：📉 深盤疲勞鎖 (Deep Impact)
    5. V6.8 保留：💰 聰明錢與虛火過濾 (已修復 Log 顯示)
    6. V6.0 保留：⚓ 市場共振 (歐亞/凱利雙向/大小)
    """
    def __init__(self):
        self.log = []
        self.flags = {
            'veto_triggered': False,
            'veto_msg': "",
            'is_panic_exemption_triggered': False
        }

    def deliberate(self, h_data, a_data, odds_data, env_data):
        h_corr = 0.0
        a_corr = 0.0
        strategy_tag = "V6.9 綜合邏輯"
        
        # --- 數據解包 ---
        h_inj = h_data.get('injury_penalty', 0)
        a_inj = a_data.get('injury_penalty', 0)
        h_mom = h_data.get('momentum', 0)
        a_mom = a_data.get('momentum', 0)
        h_fatigue = h_data.get('fatigue_days', 7)
        a_fatigue = a_data.get('fatigue_days', 7)
        
        # [V6.9] 讀取攻防數據
        h_conceded = h_data.get('conceded_avg', 1.0)
        a_conceded = a_data.get('conceded_avg', 1.0)
        # 👇 [新增] 讀取進攻數據 (預設 1.5 避免誤殺)
        h_scored = h_data.get('scored_avg', 1.5)
        a_scored = a_data.get('scored_avg', 1.5)
        
        rise_h = odds_data.get('rise_home', 0)
        rise_a = odds_data.get('rise_away', 0)
        
        h_win_rate = h_data.get('win_rate', 0)
        a_win_rate = a_data.get('win_rate', 0)
        
        handicap = env_data.get('handicap', 0)
        has_nemesis = env_data.get('nemesis', False)
        
        base_score_diff = abs(h_mom - a_mom) 

        # =================================================================
        # 1. 紅線衛兵 (RedGuard)
        # =================================================================
        CATASTROPHIC_INJURY = 25.0 
        h_critical = (h_inj >= CATASTROPHIC_INJURY)
        a_critical = (a_inj >= CATASTROPHIC_INJURY)

        if h_critical: self.log.append(f"🚑 [紅線] 主隊傷停災難({h_inj:.1f})，戰力重創")
        if a_critical: self.log.append(f"🚑 [紅線] 客隊傷停災難({a_inj:.1f})，戰力重創")

        # =================================================================
        # 1.5 ⛔ [V6.9.1 新增] 進攻死線 (Offensive Dead-Line)
        # =================================================================
        # 邏輯：場均入球 < 0.6 (如比薩) 直接判死刑，並標記為 "Dead"
        DEAD_LINE = 0.6
        h_dead = False
        a_dead = False

        if h_scored < DEAD_LINE:
            h_corr -= 15.0
            h_dead = True # 標記主隊啞火
            self.log.append(f"⛔ [進攻啞火] 主隊場均入球僅 {h_scored:.2f}，嚴重扣分 -15.0")
        
        if a_scored < DEAD_LINE:
            a_corr -= 15.0
            a_dead = True # 標記客隊啞火
            self.log.append(f"⛔ [進攻啞火] 客隊場均入球僅 {a_scored:.2f}，嚴重扣分 -15.0")

        # =================================================================
        # 2. 🛡️ [V6.9] 防火牆協議 (Firewall Protocol)
        # =================================================================
        LEAK_THRESHOLD = 1.8
        h_leak = h_conceded > LEAK_THRESHOLD
        a_leak = a_conceded > LEAK_THRESHOLD
        
        if h_leak: 
            h_corr -= 5.0
            self.log.append(f"🧱 [防守漏水] 主隊場均失球{h_conceded:.1f}，基本面扣分-5.0")
        if a_leak: 
            a_corr -= 5.0
            self.log.append(f"🧱 [防守漏水] 客隊場均失球{a_conceded:.1f}，基本面扣分-5.0")

        # =================================================================
        # 3. 聰明錢 (Smart Money) - [已修復 Log 顯示]
        # =================================================================
        SMART_RISE = 0.06 
        money_bonus = 10.0
        if base_score_diff < 10: money_bonus = 5.0 

        # [V6.9] 若防守漏水或進攻啞火，資金權重強制減半
        if h_leak or h_dead: money_bonus *= 0.5
        if a_leak or a_dead: money_bonus *= 0.5

        # 資金流向主隊 (客賠升)
        if rise_a > SMART_RISE and not h_critical and not has_nemesis:
            # 👇 修正：這裡要有縮排，並且要有內容
            if h_leak or h_dead: 
                self.log.append(f"⚠️ [資金虛火] 主隊基本面崩壞(漏水/啞火)，聰明錢權重減半") 
            
            h_corr += money_bonus
            self.log.append(f"💰 [聰明錢] 資金流向主隊，修正+{money_bonus:.1f}")
            strategy_tag = "資金流向"

        # 資金流向客隊 (主賠升)
        if rise_h > SMART_RISE and not a_critical and not has_nemesis:
            # 👇 修正：這裡要有縮排，並且要有內容
            if a_leak or a_dead: 
                self.log.append(f"⚠️ [資金虛火] 客隊基本面崩壞(漏水/啞火)，聰明錢權重減半")
            
            a_corr += money_bonus
            self.log.append(f"💰 [聰明錢] 資金流向客隊，修正+{money_bonus:.1f}")
            strategy_tag = "資金流向"


        # =================================================================
        # 4. 🎭 [V6.9.1 優化] 傷情虛實校驗 (Phantom Injury Check)
        # =================================================================
        # 邏輯：傷病重但盤口硬 -> 回補。但如果進攻啞火 (h_dead)，禁止回補！
        
        if h_inj > 15.0 and abs(handicap) < 0.5 and rise_h < 0.05:
            if not h_dead: # 只有進攻正常才回補
                refund = h_inj * 0.6 
                h_corr += refund
                self.log.append(f"🎭 [傷情虛實] 主傷重但盤口硬，莊家不懼，回補+{refund:.1f}")
            else:
                self.log.append(f"💀 [傷情虛實] 主隊進攻啞火，拒絕回補傷病分！")
            
        if a_inj > 15.0 and abs(handicap) < 0.5 and rise_a < 0.05:
            if not a_dead:
                refund = a_inj * 0.6
                a_corr += refund
                self.log.append(f"🎭 [傷情虛實] 客傷重但盤口硬，莊家不懼，回補+{refund:.1f}")
            else:
                self.log.append(f"💀 [傷情虛實] 客隊進攻啞火，拒絕回補傷病分！")

        # =================================================================
        # 5. [V6.8] 深盤疲勞鎖 (Deep Impact)
        # =================================================================
        DEEP_HANDICAP_THRESHOLD = 1.25
        FATIGUE_LIMIT = 4
        FATIGUE_PENALTY = -12.0
        
        if handicap > DEEP_HANDICAP_THRESHOLD and h_fatigue <= FATIGUE_LIMIT:
            h_corr += FATIGUE_PENALTY
            self.log.append(f"📉 [深盤疲勞] 主讓深盤但休{h_fatigue}天，修正{FATIGUE_PENALTY}")
        elif handicap < -DEEP_HANDICAP_THRESHOLD and a_fatigue <= FATIGUE_LIMIT:
            a_corr += FATIGUE_PENALTY
            self.log.append(f"📉 [深盤疲勞] 客讓深盤但休{a_fatigue}天，修正{FATIGUE_PENALTY}")

        # =================================================================
        # 6. 市場敬畏 (Market Respect)
        # =================================================================
        if h_mom > a_mom + 15 and rise_h > 0.08:
            penalty = -20.0
            h_corr += penalty
            self.flags['veto_triggered'] = True
            self.flags['veto_msg'] = "主隊遭市場拋售"
            self.log.append(f"📉 [市場敬畏] 主隊遭拋售，修正{penalty}")

        if a_mom > h_mom + 15 and rise_a > 0.08:
            penalty = -20.0
            a_corr += penalty
            self.flags['veto_triggered'] = True
            self.flags['veto_msg'] = "客隊遭市場拋售"
            self.log.append(f"📉 [市場敬畏] 客隊遭拋售，修正{penalty}")

        # =================================================================
        # 7. [V5.1] H2H 盤路回測修正
        # =================================================================
        backtest = env_data.get('h2h_backtest', {})
        bt_rate = backtest.get('backtest_win_rate', 0.5)
        bt_total = backtest.get('total', 0)
        current_hdp = env_data.get('handicap', 0)
        
        if bt_total >= 3:
            if current_hdp >= 1.0 and bt_rate >= 0.7:
                bonus = 20.0
                h_corr += bonus
                self.log.append(f"🛡️ [盤路回測] 深盤阻力生效！歷史受讓({current_hdp})不敗率{bt_rate:.0%}，主修正+{bonus}")
                strategy_tag = "盤路回測"
            elif current_hdp <= -1.0 and bt_rate <= 0.3:
                penalty = -20.0
                h_corr += penalty 
                self.log.append(f"📉 [盤路回測] 穿盤能力不足！歷史讓球({current_hdp})贏盤率僅{bt_rate:.0%}，主修正{penalty}")

        # =================================================================
        # 8. [V6.7] 傷停回補 (Bench Depth)
        # =================================================================
        if h_inj > 15.0 and h_win_rate > 0.5 and not h_dead: # 進攻啞火也不給板凳分
            h_corr += 5.0
            self.log.append("🧬 [板凳深度] 強隊傷停適應，回補+5.0")
            
        if a_inj > 15.0 and a_win_rate > 0.5 and not a_dead:
            a_corr += 5.0
            self.log.append("🧬 [板凳深度] 強隊傷停適應，回補+5.0")

        # =================================================================
        # 9. [V6.0] 歐亞錨定、大小球共振、凱利風控 (雙向版)
        # =================================================================
        market = env_data.get('market_resonance', {})
        theo_diff = market.get('theo_diff', 0)
        ou_support = market.get('ou_support', 'Neutral')
        kelly_sig = market.get('kelly_signal', 'Neutral')
        
        # --- A. 歐亞錨定 ---
        if theo_diff >= 0.5:
            penalty = -15.0
            if handicap > 0: h_corr += penalty
            elif handicap < 0: a_corr += penalty
            self.log.append(f"⚓ [歐亞陷阱] 歐賠支撐不足，修正{penalty}")
        elif theo_diff <= -0.5:
            bonus = 12.0
            if handicap > 0: h_corr += bonus
            elif handicap < 0: a_corr += bonus
            self.log.append(f"🛡️ [莊家信心] 亞盤深於歐賠，修正+{bonus}")

        # --- B. 大小球共振 ---
        if ou_support == 'Home':
            h_corr += 8.0
            self.log.append(f"🌊 [大小共振] 大球利好主隊，修正+8.0")
        elif ou_support == 'Away':
            a_corr += 8.0
            self.log.append(f"🌊 [大小共振] 大球利好客隊，修正+8.0")
            
        # --- C. 凱利風控 (雙向監控) ---
        # 1. 主隊信號
        if kelly_sig in ['Home_Guard', 'Guard']: 
            h_corr += 10.0
            self.log.append("💰 [凱利防範] 主勝防範，修正+10.0")
        elif kelly_sig in ['Home_SuperGuard', 'SuperGuard']:
            h_corr += 15.0
            self.log.append("💰 [凱利鐵鎖] 主勝極度防範，修正+15.0")
        elif kelly_sig in ['Home_Trap', 'Trap']:
            h_corr -= 12.0
            self.log.append("🚨 [凱利誘盤] 主勝誘盤，修正-12.0")

        # 2. 客隊信號 (新增！)
        if kelly_sig == 'Away_Guard':
            a_corr += 10.0
            self.log.append("💰 [凱利防範] 客勝防範，修正+10.0")
        elif kelly_sig == 'Away_SuperGuard':
            a_corr += 15.0
            self.log.append("💰 [凱利鐵鎖] 客勝極度防範，修正+15.0")
        elif kelly_sig == 'Away_Trap':
            a_corr -= 12.0
            self.log.append("🚨 [凱利誘盤] 客勝誘盤，修正-12.0")

        # =================================================================
        # 10. 🛠️ [V6.0] 額外保險
        # =================================================================
        h_corr = round(h_corr, 2)
        a_corr = round(a_corr, 2)

        return h_corr, a_corr, self.log, strategy_tag

# =================================================================
# 🛡️ V3.7 核心架構：三權分立校驗閥 (Tri-Check Validator)
# =================================================================
class PrecisionValidatorV50_Ultimate:
    """
    V5.0 Ultimate Final: 全能上帝版 (Full Stack Integration)
    
    【融合版本歷史】
    - V3.7.3: 資金流向 (Pin Change) + 頹勢安全鎖 (Cold Streak Lock)
    - V3.8/3.9: 完整傷病權重字典 + H2H 天敵 (Nemesis)
    - V3.9.2: 趁火打劫 (Counter-Strike) 機制
    - V4.0: 盤路解碼 (Handicap Decoder) + 毒藥過濾 (Poison Filter) + 盤路加成
    - V4.1: 狙擊手模式 (Sniper): 客場蟲、防守漏水、欺善怕惡
    - V5.0: 上帝模式 (God Mode): 物理傷害絕對壓制 (Override)
    """
    
    @staticmethod
    def validate_decision(
        match_data: dict, 
        base_score_diff: float,
        odds_trend: dict,
        risk_level: str
    ) -> dict:
        
        confidence = 0.0
        decision_log = []
        status = "SKIP"
        
        # ==========================================================
        # 0. 數據全維度提取 & 預處理 (Data Extraction)
        # ==========================================================
        
        # [基礎方向]
        fundamental_dir = "HOME" if base_score_diff > 0 else "AWAY"
        fundamental_strength = abs(base_score_diff)
        pin_chg = odds_trend.get('pin_change', 0.0)
        
        # [V3.8 完整傷病字典]
        def calculate_injury_score(text):
            if not text: return 0
            score = 0
            weights = {
                '十字韌帶': 15, '阿基里斯': 15, '賽季報銷': 15, '骨折': 12, 
                '斷裂': 12, '手術': 12, '重傷': 12, '撕裂': 8, 
                '半月板': 8, '缺陣': 2, '停賽': 3, '國家隊': 2, '發炎': 2
            }
            for keyword, weight in weights.items():
                score += text.count(keyword) * weight
            return score

        h_inj_text = match_data.get('home_injury_text', '')
        a_inj_text = match_data.get('away_injury_text', '')
        h_disaster_score = calculate_injury_score(h_inj_text)
        a_disaster_score = calculate_injury_score(a_inj_text)
        
        # [V4.0 盤路解碼與近況]
        h_raw_form = match_data.get('home_recent_form', [])
        a_raw_form = match_data.get('away_recent_form', [])
        
        def get_handicap_rate(form_data):
            text = str(form_data)
            win = text.count('贏') + text.count('赢')
            loss = text.count('輸') + text.count('输')
            total = win + loss + text.count('走')
            return win / total if total > 0 else 0.5

        h_handicap_rate = get_handicap_rate(h_raw_form)
        a_handicap_rate = get_handicap_rate(a_raw_form)
        
        # 提取勝場數 (用於 V3.7 頹勢鎖)
        h_wins = str(h_raw_form).count('W')
        a_wins = str(a_raw_form).count('W')

        # [V4.1 深度數據]
        home_stats = match_data.get('home_stats', {})
        away_stats = match_data.get('away_stats', {})
        
        a_away_win_rate = away_stats.get('away_win_rate', 0.11) # 客隊客場勝率
        h_conceded = home_stats.get('conceded_avg', 1.0)
        a_conceded = away_stats.get('conceded_avg', 2.4)
        h_goals = home_stats.get('goals_scored', 20)
        a_goals = away_stats.get('goals_scored', 26)
        
        # [V3.8 H2H 天敵數據]
        h2h_form = match_data.get('h2h_recent_form', [])
        h2h_wins = str(h2h_form).count('W')
        # 定義：至少3場交手且主隊0勝
        is_h2h_nemesis = (len(h2h_form) >= 3 and h2h_wins == 0)

        # ==========================================================
        # 👑 第一層：God Mode (V5.0 人性化直覺 - 物理壓制)
        # ==========================================================
        # 這是最高優先級，若觸發，直接 Return，無視後面所有邏輯 (包括客場蟲、防守差等)
        
        CRITICAL_INJURY = 30 # 定義殘廢線
        
        # [場景 A] 主隊殘廢 (V5.0 Override)
        if h_disaster_score >= CRITICAL_INJURY:
            decision_log.append(f"🚑 [結構崩壞] 主隊傷病分({h_disaster_score})爆表")
            # V5.0 核心：無視客隊爛數據，只要客隊有進攻能力或盤路尚可
            if a_goals >= h_goals or a_handicap_rate >= 0.3:
                return {
                    'status': "BET_AWAY",
                    'confidence': 0.92,
                    'log': f"🔥 [人性直覺] 主隊殘廢，無視客隊客場劣績，強制推薦客勝 | {decision_log[0]}"
                }
            else:
                decision_log.append("⚠️ 客隊進攻太弱，可能無法利用主隊傷病")
        
        # [場景 B] 客隊殘廢 (V5.0 Override)
        if a_disaster_score >= CRITICAL_INJURY:
            decision_log.append(f"🚑 [結構崩壞] 客隊傷病分({a_disaster_score})爆表")
            if h_goals >= a_goals or h_handicap_rate >= 0.3:
                return {
                    'status': "BET_HOME",
                    'confidence': 0.92,
                    'log': f"🔥 [人性直覺] 客隊殘廢，無視主隊近況差，強制推薦主勝 | {decision_log[0]}"
                }
            else:
                 decision_log.append("⚠️ 主隊進攻太弱，可能無法利用客隊傷病")

        # ==========================================================
        # 🛑 第二層：硬性過濾 (V3.8 天敵 & V4.0 盤路毒藥)
        # ==========================================================
        # 如果沒有觸發 God Mode，我們必須執行嚴格的風控
        
        # 1. 天敵檢查 (V3.8)
        if fundamental_dir == "HOME" and is_h2h_nemesis:
            return {'status': "SKIP", 'confidence': 0, 'log': f"🛑 [天敵紅線] 主隊遇剋星(近{len(h2h_form)}場0勝)"}
            
        # 2. 盤路毒藥 (V4.0) - 贏盤率 < 20% 直接殺
        if fundamental_dir == "HOME" and h_handicap_rate <= 0.2:
            return {'status': "SKIP", 'confidence': 0, 'log': f"🛑 [盤路毒藥] 主隊贏盤率極低({h_handicap_rate:.0%})"}
        
        if fundamental_dir == "AWAY" and a_handicap_rate <= 0.2:
            return {'status': "SKIP", 'confidence': 0, 'log': f"🛑 [盤路毒藥] 客隊贏盤率極低({a_handicap_rate:.0%})"}

        # ==========================================================
        # 🎯 第三層：Sniper Mode (V4.1 弱點狙擊 & V3.9.2 趁火打劫)
        # ==========================================================
        
        sniper_penalty = 0
        
        # 1. 客場蟲檢查 (V4.1)
        if fundamental_dir == "AWAY" and a_away_win_rate < 0.15:
            decision_log.append(f"⚠️ [客場蟲] 客勝率僅 {a_away_win_rate:.0%}")
            sniper_penalty -= 20
            
        # 2. 防守漏水檢查 (V4.1)
        if fundamental_dir == "AWAY" and a_conceded > 2.0:
            decision_log.append(f"⚠️ [防守漏水] 客隊場均失球 {a_conceded}")
            sniper_penalty -= 15
        elif fundamental_dir == "HOME" and h_conceded > 2.0:
            decision_log.append(f"⚠️ [防守漏水] 主隊場均失球 {h_conceded}")
            sniper_penalty -= 15
            
        # 3. 欺善怕惡 (V4.1 Bully Factor)
        # 假設傳入數據中有 opponent_rank，若無則默認 9 (中游)
        opponent_rank = match_data.get('opponent_rank', 9) 
        if fundamental_dir == "HOME" and opponent_rank <= 9 and h_wins == 0: 
             decision_log.append(f"⚠️ [遇強即死] 主隊對陣強隊無勝績")
             sniper_penalty -= 10

        # 4. 趁火打劫 (V3.9.2) - 中等傷病加分 (25-30分)
        # 註：>30分已在第一層 God Mode 處理，這裡是處理 "嚴重但未致殘" 的情況
        if h_disaster_score >= 25 and h_disaster_score < 30:
            decision_log.append(f"🚑 [重傷] 主隊傷病嚴重({h_disaster_score})")
            if fundamental_dir == "AWAY": sniper_penalty += 15 # 給客隊加分
            
        if a_disaster_score >= 25 and a_disaster_score < 30:
            decision_log.append(f"🚑 [重傷] 客隊傷病嚴重({a_disaster_score})")
            if fundamental_dir == "HOME": sniper_penalty += 15 # 給主隊加分

        # ==========================================================
        # 💰 第四層：Fund Mode (V3.7.3 資金與狀態裁決)
        # ==========================================================
        
        # 計算修正後的實力分
        final_strength = fundamental_strength + sniper_penalty
        
        market_dir = "NEUTRAL"
        NOISE_THRESHOLD = 0.05
        if pin_chg < -NOISE_THRESHOLD: market_dir = "HOME"
        elif pin_chg > NOISE_THRESHOLD: market_dir = "AWAY"
        
        decision_log.append(f"📊 修正實力: {final_strength:.1f} | 💰 資金: {market_dir}")

        # --- 最終裁決邏輯 ---
        
        # A. 共振 (Resonance)
        if fundamental_dir == market_dir:
            if final_strength > 10: 
                status = f"BET_{fundamental_dir}"
                confidence = 0.85
                # [V4.0] 盤路加成 (Bonus)
                if (fundamental_dir == "HOME" and h_handicap_rate > 0.6) or \
                   (fundamental_dir == "AWAY" and a_handicap_rate > 0.6):
                    confidence += 0.05
                    decision_log.append("✅ [完美共振+盤路強勢]")
                else:
                    decision_log.append("✅ [完美共振]")
            else:
                decision_log.append("⚠️ [優勢不足] 扣除弱點後分數過低")

        # B. 資金中立
        elif market_dir == "NEUTRAL":
            if final_strength > 15:
                status = f"BET_{fundamental_dir}"
                confidence = 0.75
                decision_log.append("✅ [單核驅動] 信賴修正後的數據")
            else:
                decision_log.append("⚠️ [分數不足]")

        # C. 資金逆勢 (Risk)
        else:
            if abs(pin_chg) >= 0.15:
                status = "SKIP"
                decision_log.append("🛑 [市場否決] 資金大幅逆勢")
            else:
                # 🚨 [V3.7.3] 頹勢安全鎖 (Cold Streak Lock)
                # 即使數據再好，如果近況 0 勝且資金逆勢，絕對不買
                if fundamental_dir == "HOME" and h_wins == 0:
                    status = "SKIP"
                    decision_log.append(f"🚫 [狀態崩盤] 主隊近況0勝且資金逆勢，禁止接飛刀")
                
                elif fundamental_dir == "AWAY" and a_wins == 0:
                    status = "SKIP"
                    decision_log.append(f"🚫 [狀態崩盤] 客隊近況0勝且資金逆勢，禁止接飛刀")
                
                else:
                    # 抗壓條件：盤路必須好 (>50%) 且 分數夠高
                    handicap_ok = (fundamental_dir == "HOME" and h_handicap_rate > 0.5) or \
                                  (fundamental_dir == "AWAY" and a_handicap_rate > 0.5)
                    
                    if final_strength > 20 and handicap_ok:
                        status = f"BET_{fundamental_dir}"
                        confidence = 0.65
                        decision_log.append("⚠️ [抗壓出擊] 實力強勁且盤路佳，無視資金微逆")
                    else:
                        status = "SKIP"
                        decision_log.append("🚫 [動能不足] 無法抵消資金逆勢")

        # 風控攔截
        if risk_level == '🔴 極高風險' and status != "SKIP":
            status = "SKIP"
            decision_log.append("🛑 [風控攔截]")

        return {
            'status': status,
            'confidence': min(confidence, 0.95), # 封頂 0.95
            'log': " | ".join(decision_log)
        }


# ═══════════════════════════════════════════════════════════════
# 【模組 9】主系統類別 SmartBettingSystemV293 (Smart Exemption)
# ═══════════════════════════════════════════════════════════════
class SmartBettingSystemV293:
    """"🚀 V2.9.7 Alpha 智能投注系統 (Trap & Casualty Edition)"""
    
        def __init__(self, enable_poisson=True, bankroll=10000):
        self.enable_poisson = enable_poisson
        self.bankroll = bankroll
        self.handicap_dict = HandicapDictionary()
        self.league_adapter = LeagueCharacteristicsAdapterV268()
        self.smart_no_recommendation = SmartNoRecommendationSystem()
        self.kelly_optimizer = KellyOptimizer227()
        self.risk_preference = 'moderate'
        self.home_away_analyzer = HomeAwayDetailAnalyzer()
        self.handicap_history_analyzer = HandicapHistoryAnalyzer()
        self.h2h_deep_analyzer = H2HDeepAnalyzer()
        self.risk_evaluator = MultiDimensionalRiskEvaluator()
        
        # ══════════════════════════════════════════════════════
        # 🚀 [V2.9.7 Alpha] 新增模組初始化 (請確保這三行在最後)
        # ══════════════════════════════════════════════════════
        self.lineup_analyzer =      LineupImpactAnalyzer()
        self.dealer_engine = DealerPsychologyEngine()
        self.trend_analyzer = SpecificHandicapTrendAnalyzer()


    def _poisson_probability(self, actual, mean):
        return math.exp(-mean) * (mean**actual) / math.factorial(actual)

    def calculate_handicap_coverage(self, h_exp, a_exp, handicap_line):
        cover_prob = 0.0
        for h in range(7):
            for a in range(7):
                prob_score = self._poisson_probability(h, h_exp) * self._poisson_probability(a, a_exp)
                if (h - a) > handicap_line:
                    cover_prob += prob_score
        return cover_prob * 100

    def calculate_mean_reversion(self, recent_form_list):
        if not recent_form_list or len(recent_form_list) < 3: return 0
        streak = 0
        last_result = recent_form_list[0]
        for res in recent_form_list[:5]:
            if res == last_result: streak += 1
            else: break
        score_adjust = 0
        if streak >= 4:
            if last_result == 'W': score_adjust = -1.5 * (streak - 3)
            elif last_result == 'L': score_adjust = 1.0 * (streak - 3)
        return score_adjust

    def calculate_style_mismatch(self, h_goals_for, h_goals_against, a_goals_for, a_goals_against):
        h_def = max(0.5, h_goals_against)
        a_def = max(0.5, a_goals_against)
        home_attack_ratio = h_goals_for / a_def
        away_attack_ratio = a_goals_for / h_def
        h_bonus = 0; a_bonus = 0
        if home_attack_ratio > 1.5: h_bonus += 2.0
        elif home_attack_ratio < 0.7: h_bonus -= 1.5
        if away_attack_ratio > 1.5: a_bonus += 2.0
        elif away_attack_ratio < 0.7: a_bonus -= 1.5
        return h_bonus, a_bonus

    def _calculate_opponent_adjustment(self, home_rank, away_rank):
        diff = away_rank - home_rank
        if diff >= 10: h_adj = 1.3
        elif diff >= 5: h_adj = 1.15
        elif diff <= -10: h_adj = 0.7
        elif diff <= -5: h_adj = 0.85
        else: h_adj = 1.0
        diff_a = home_rank - away_rank
        if diff_a >= 10: a_adj = 1.3
        elif diff_a >= 5: a_adj = 1.15
        elif diff_a <= -10: a_adj = 0.7
        elif diff_a <= -5: a_adj = 0.85
        else: a_adj = 1.0
        return h_adj, a_adj

    def _poisson_analysis(self, h_avg, a_avg, h_conc, a_conc, l_avg, handicap_val=0, home_rank=10, away_rank=10):
        h_exp = h_avg * (a_conc/l_avg)
        a_exp = a_avg * (h_conc/l_avg)
        h_adj, a_adj = self._calculate_opponent_adjustment(home_rank, away_rank)
        h_exp *= h_adj; a_exp *= a_adj
        if abs(handicap_val) > 1.75:
            if handicap_val > 0: h_exp *= 0.85
            else: a_exp *= 0.85
        score_probs = {}
        for h in range(6):
            for a in range(6):
                p = stats.poisson.pmf(h, h_exp) * stats.poisson.pmf(a, a_exp)
                score_probs[(h,a)] = p
        heavy_defeat = sum(p for (h,a), p in score_probs.items() if abs(h-a) >= 3)
        return {'home_expected_goals': h_exp, 'away_expected_goals': a_exp, 'score_probabilities': score_probs, 'heavy_defeat_risk': heavy_defeat}
    
    def _normalize_handicap_diff(self, target_hdp: float, ref_hdp: float, ref_odds: float = None, home_rank: int = 0, away_rank: int = 0) -> Tuple[float, str, bool]:
        if ref_hdp is None or target_hdp == ref_hdp: return 0.0, "", False
        
        diff = abs(ref_hdp) - abs(target_hdp)
        correction = 0.0; msg = ""; ban_triggered = False
        
        is_home_rel_weak = (home_rank - away_rank) >= 4
        is_away_rel_weak = (away_rank - home_rank) >= 4
        
        if diff > 0.1: 
            if (target_hdp > 0 and is_home_rel_weak) or (target_hdp < 0 and is_away_rel_weak):
                correction = 0.0; ban_triggered = True
                msg = f"⚓ [錨定禁令] 讓球方相對弱勢(Rank差>4)，Pin深盤視為誘盤，取消加分"
            else:
                correction = 12.0
                msg = f"⚓ [盤口錨定] Pin盤({ref_hdp})較深，本盤({target_hdp})門檻低具優勢"
        elif diff < -0.1: 
            correction = -15.0
            msg = f"⚠️ [盤口錨定] Pin盤({ref_hdp})較淺，本盤({target_hdp})過度強勢需防冷"
        else:
            correction = 0.0
            msg = f"⚓ [盤口錨定] 盤口一致(Diff:{diff:.2f})，無修正"
            
        return correction, msg, ban_triggered

    def _detect_defense_collapse_v223(self, team, avg_conc, form, match_type):
        if match_type == 'Playoff' and avg_conc < 2.5: return {'level': '🟢 季後賽豁免', 'score_adjustment': 0}
        if avg_conc >= 2.0: return {'level': '🔴 防守崩潰', 'score_adjustment': -20}
        return {'level': '🟢 正常', 'score_adjustment': 0}

    # 1. 修改函數定義，加入 ai_injury_feed 參數 (預設為 None)
    def analyze_match(self, match_data: Dict, ai_injury_feed: str = None) -> Dict:
        
        # 2. 呼叫剛才修復好的 DataInjector (解決 75分 vs 45分 問題)
        # 這裡會自動讀取 raw_text 裡面的 # Pin 賠率、HKJC 盤口等
        if 'raw_text' in match_data:
            match_data = DataInjector.inject_manual_data(match_data['raw_text'], match_data)

        # 3. 處理 AI 傷停注入 (解決圖片讀取問題)
        if ai_injury_feed:
            current_h_inj = match_data.get('home_injury_text')
            # 邏輯：如果你手動無寫傷停，或者寫 "無"，系統就會自動填入 AI 讀到的內容
            if not current_h_inj or current_h_inj == "無":
                print(f"🤖 [系統] 注入 AI 讀圖傷停資訊: {ai_injury_feed}")
                match_data['home_injury_text'] = ai_injury_feed
                match_data['away_injury_text'] = ai_injury_feed

        # 4. [V2.9.3 Patch] 數據源標準化 (接駁回你原本的邏輯)
        match_data = DataNormalizer.normalize_relative_scores(match_data)
        
        # 1. 數據驗證
        match_data = DataValidator.validate_match_data(match_data)
        match_type = match_data.get('match_type', 'Regular')
        for kw in ['季後賽', 'Playoff', 'Cup', 'Final', '墨西聯附']:
            if kw in match_data.get('league', ''): match_type = 'Playoff'; break
        match_data['match_type'] = match_type

        # 2. 賠率處理
        if 'company_odds' in match_data and match_data['company_odds']:
            try:
                cm = CompanyOddsManagerV229(match_data['company_odds'])
                best = cm.get_best_odds()
                match_data.update({'home_odds': best['home_odds'], 'away_odds': best['away_odds'], 'odds_source_name': best['source_name'], 'odds_change': best['odds_change'], 'all_companies_data': best['all_companies_data']})
            except: pass

        # 3. 盤口識別
        handicap_info = self.handicap_dict.identify_handicap(match_data['handicap'])
        target_hdp_val = handicap_info['value'] 
        pin_hdp_input = match_data.get('pin_handicap', match_data['handicap'])
        pin_hdp_val = self.handicap_dict.identify_handicap(pin_hdp_input)['value']
        b365_hdp_input = match_data.get('b365_handicap', match_data['handicap'])
        b365_hdp_val = self.handicap_dict.identify_handicap(b365_hdp_input)['value']

        league_info = self.league_adapter.get_league_adjustments(match_data.get('league', 'DEFAULT'))
        home_ranking = match_data['home_ranking']; away_ranking = match_data['away_ranking']
        
        # 4. 基礎評分
        home_perf = self.home_away_analyzer.analyze_home_performance(match_data.get('home_home_matches', 0), match_data.get('home_home_wins', 0), 0, 0, match_data.get('home_home_goals_for', 0), 0)
        away_perf = self.home_away_analyzer.analyze_away_performance(match_data.get('away_away_matches', 0), match_data.get('away_away_wins', 0), 0, 0, match_data.get('away_away_goals_for', 0), 0)
        home_hdp_perf = self.handicap_history_analyzer.analyze_handicap_performance(match_data.get('home_handicap_history', {}), handicap_info['value'])
        away_hdp_perf = self.handicap_history_analyzer.analyze_handicap_performance(match_data.get('away_handicap_history', {}), -handicap_info['value'])

        home_base_score = home_perf['home_advantage_score'] if home_perf['home_advantage_score'] > 50 else (20 - home_ranking) * 5 + league_info['home_bonus']
        away_base_score = away_perf['away_strength_score'] if away_perf['away_strength_score'] > 50 else (20 - away_ranking) * 5
        home_form_score = 70; away_form_score = 70 
        
        # 5. 泊松與防守崩潰
        poisson_result = None
        if self.enable_poisson:
            poisson_result = self._poisson_analysis(match_data.get('home_goals_scored', 0)/5.0, match_data.get('away_goals_scored', 0)/5.0, match_data.get('home_goals_conceded', 0)/5.0, match_data.get('away_goals_conceded', 0)/5.0, league_info['avg_goals'], handicap_info['value'], home_ranking, away_ranking)

        home_collapse = self._detect_defense_collapse_v223(match_data['home_team'], match_data.get('home_goals_conceded', 0)/5.0, match_data['home_recent_form'], match_type)
        away_collapse = self._detect_defense_collapse_v223(match_data['away_team'], match_data.get('away_goals_conceded', 0)/5.0, match_data['away_recent_form'], match_type)
        home_bonus = league_info['home_bonus']
        
        h2h_deep = self.h2h_deep_analyzer.analyze_h2h_handicap(match_data.get('h2h_details', []), handicap_info['value'])
        # 👇 [V5.1 新增] 呼叫回測引擎
        h2h_backtest = self.h2h_deep_analyzer.simulate_handicap_history(match_data.get('h2h_details', []), handicap_info['value'])
        
        # [V2.9.3] 強制 H2H 剋星修正
        if 'h2h_recent_form' in match_data:
            h2h_form_raw = match_data['h2h_recent_form']
            h2h_str = "".join(h2h_form_raw).upper() if isinstance(h2h_form_raw, list) else str(h2h_form_raw).upper()
            valid_h2h = [c for c in h2h_str if c in ['W', 'D', 'L']]
            if valid_h2h:
                h_total = len(valid_h2h); h_wins = valid_h2h.count('W')
                h_rate = h_wins / h_total
                if h_rate >= 0.6: h2h_deep['nemesis_relationship'] = f'主隊明顯剋星 (近{h_total}贏{h_wins})'
                elif h_rate <= 0.2: h2h_deep['nemesis_relationship'] = f'客隊明顯剋星 (近{h_total}輸{h_total-h_wins})'
                else: h2h_deep['nemesis_relationship'] = '互有勝負'

        home_total_score = home_base_score * 0.3 + home_form_score * 0.3 + 70 * 0.2 + home_bonus
        away_total_score = away_base_score * 0.3 + away_form_score * 0.3 + 70 * 0.2
        home_total_score += (home_hdp_perf['adaptation_score'] - 50) * 0.1
        away_total_score += (away_hdp_perf['adaptation_score'] - 50) * 0.1
        
        if '主隊明顯剋星' in h2h_deep['nemesis_relationship']: home_total_score += 5
        elif '客隊明顯剋星' in h2h_deep['nemesis_relationship']: away_total_score += 5

        # 高階數據處理
        h_detailed = match_data.get('home_recent_matches_detailed', [])
        a_detailed = match_data.get('away_recent_matches_detailed', [])
        curr_date = datetime.now()
        h_sorted = AdvancedMetrics.sort_matches_by_date(h_detailed, curr_date)
        a_sorted = AdvancedMetrics.sort_matches_by_date(a_detailed, curr_date)

        h_goals_seq = []
        for m in h_sorted:
            try:
                s = m.get('score', '0-0').split('-')
                if m.get('venue', 'home') == 'home': h_goals_seq.append(int(s[0]))
                else: h_goals_seq.append(int(s[1]))
            except: pass
        a_goals_seq = []
        for m in a_sorted:
            try:
                s = m.get('score', '0-0').split('-')
                if m.get('venue', 'home') == 'home': a_goals_seq.append(int(s[0]))
                else: a_goals_seq.append(int(s[1]))
            except: pass

        h_volatility = AdvancedMetrics.calculate_volatility(h_goals_seq)
        a_volatility = AdvancedMetrics.calculate_volatility(a_goals_seq)
        h_time_decay = AdvancedMetrics.calculate_time_decay(h_sorted)
        a_time_decay = AdvancedMetrics.calculate_time_decay(a_sorted)
        h_fatigue = AdvancedMetrics.calculate_fatigue(h_sorted, curr_date)
        a_fatigue = AdvancedMetrics.calculate_fatigue(a_sorted, curr_date)

        h_recent_rev = match_data.get('home_recent_form', [])
        a_recent_rev = match_data.get('away_recent_form', [])
        h_mom_val = AdvancedMetrics.calculate_weighted_momentum(h_recent_rev)
        a_mom_val = AdvancedMetrics.calculate_weighted_momentum(a_recent_rev)
        mom_diff_val = h_mom_val - a_mom_val
        is_high_momentum = abs(mom_diff_val) >= 20.0 

        # ✅ 變數初始化區
        correction_msg = []; home_correction = 0; away_correction = 0
        force_no_recommend = False; dealer_killer_triggered = False
        veto_triggered = False; veto_msg = "無"
        is_anchor_ban_triggered = False
        has_nemesis_exemption = False; nemesis_type = h2h_deep.get('nemesis_relationship', '')
        match_data['forced_draw_risk_increase'] = False 
        strategy_used = "🧠 V2.9.9 綜合動態運算"
        
        home_odds = match_data.get('home_odds', 0); away_odds = match_data.get('away_odds', 0)
        handicap_val = handicap_info.get('value', 0); abs_hdp = abs(handicap_val)
        
        pin_data = match_data.get('company_odds', {}).get('PIN', {})
        b365_data = match_data.get('company_odds', {}).get('B365', {})
        pin_chg_h = (pin_data.get('current_home', 0) - pin_data.get('early_home', 0)) if pin_data.get('early_home') else 0
        b365_chg_h = (b365_data.get('current_home', 0) - b365_data.get('early_home', 0)) if b365_data.get('early_home') else 0
        max_drop_home = min(pin_chg_h, b365_chg_h); rise_home = max(pin_chg_h, b365_chg_h)
        pin_chg_a = (pin_data.get('current_away', 0) - pin_data.get('early_away', 0)) if pin_data.get('early_away') else 0
        b365_chg_a = (b365_data.get('current_away', 0) - b365_data.get('early_away', 0)) if b365_data.get('early_away') else 0
        max_drop_away = min(pin_chg_a, b365_chg_a); rise_away = max(pin_chg_a, b365_chg_a)
        is_divergent = (pin_chg_h * b365_chg_h < 0) and (abs(pin_chg_h - b365_chg_h) > 0.05)

        # 🟢 [V2.9.0] 智能盤口錨定
        pin_curr_h = pin_data.get('current_home', 0)
        pin_corr, pin_msg, is_anchor_ban_triggered = self._normalize_handicap_diff(target_hdp_val, pin_hdp_val, pin_curr_h, home_ranking, away_ranking)
        if pin_corr != 0: correction_msg.append(pin_msg)
        if target_hdp_val > 0: home_correction += pin_corr
        elif target_hdp_val < 0: away_correction += pin_corr

        b365_corr, b365_msg, _ = self._normalize_handicap_diff(target_hdp_val, b365_hdp_val, 0)
        if b365_corr != 0: correction_msg.append(b365_msg)
        if target_hdp_val > 0: home_correction += (b365_corr * 0.5)
        elif target_hdp_val < 0: away_correction += (b365_corr * 0.5)

        # 傷停與莊家心理 (V2.9.7)
        h_inj_text = match_data.get('home_injury_text', '')
        a_inj_text = match_data.get('away_injury_text', '')
        
        h_inj_pen, h_inj_msg = self.lineup_analyzer.analyze_injury_impact(h_inj_text, match_data['home_team'])
        a_inj_pen, a_inj_msg = self.lineup_analyzer.analyze_injury_impact(a_inj_text, match_data['away_team'])
        
        if h_inj_pen > 0: home_correction -= h_inj_pen; correction_msg.append(h_inj_msg)
        if a_inj_pen > 0: away_correction -= a_inj_pen; correction_msg.append(a_inj_msg)

        # ═══════════════════════════════════════════════════════════════
        # ⚖️ [V3.0 架構] 召喚總法官 (Call the Final Arbiter)
        # ═══════════════════════════════════════════════════════════════
        
        # 1. 準備呈堂證供 (Prepare Evidence)
        def calculate_win_rate_helper(form_list):
            if not form_list: return 0.0
            wins = [res for res in form_list if str(res).upper() == 'W']
            return len(wins) / len(form_list)

        # =================================================================
        # 🧠 [V6.9.2 AI 數據適配核心] - 智能過濾主客場數據
        # =================================================================

        def get_venue_specific_avg(match_list, target_venue, total_goals_fallback):
            """
            從近期對戰列表中，只提取特定場地(主/客)的數據來計算平均值。
            如果列表中沒有該場地的比賽，才退而求其次使用賽季總平均。
            """
            # 1. 嘗試從詳細列表中篩選 (這是最準確的，因為反映近況且區分主客)
            relevant_matches = [m for m in match_list if m.get('venue') == target_venue]
            
            if relevant_matches:
                total_goals = 0
                for m in relevant_matches:
                    scores = m['score'].split('-')
                    # 如果是主場，取前面(0)；如果是客場，取後面(1)
                    # 注意：這裡假設 match_list 都是以 "主-客" 格式記錄比分
                    # 但你的數據源似乎是根據 venue 變動的，這裡做個防呆：
                    # 通常 score 格式是 "己方-對方" 還是 "主隊-客隊"? 
                    # 根據你的數據: vs Inter (Away) 2-6 -> 應該是 己方-對方 或 主-客?
                    # 假設你的數據格式是標準的 "HomeTeamScore-AwayTeamScore"
                    if target_venue == 'home':
                        # 我們是主隊，取前面的數字
                        total_goals += int(scores[0])
                    else:
                        # 我們是客隊，取後面的數字 (因為我們在客場)
                        total_goals += int(scores[1])
                
                return total_goals / len(relevant_matches)
            
            # 2. 如果列表中竟然沒有主場比賽，則使用賽季總數據估算
            # 假設賽季進行了一半 (約19場)，避免除以5造成的數據膨脹
            return total_goals_fallback / 19.0

        # --- 計算真實攻防數據 ---
        
        # 主隊進攻力 (只看主場比賽)
        # 比薩案例：列表有2場主場(1-1, 0-3)，入球 1+0=1。平均 0.5 -> 觸發死刑！
        h_scored_real = get_venue_specific_avg(
            match_data.get('home_recent_matches_detailed', []), 
            'home', 
            match_data.get('home_goals_scored', 0)
        )
        
        # 主隊防守力 (只看主場比賽)
        h_conceded_real = get_venue_specific_avg(
            match_data.get('home_recent_matches_detailed', []), 
            'home', # 注意這裡有點 tricky，計算主隊失球，我們還是看它的主場比賽
            match_data.get('home_goals_conceded', 0)
        )
        # 修正：上面的函數 get_venue_specific_avg 是算入球的，算失球要改一下邏輯
        # 為了代碼簡潔，我們直接在下面手寫失球邏輯

        # --- 重新編寫更直接的邏輯 (避免函數複雜化) ---
        
        # 1. 主隊進攻 (Home Scored)
        h_home_games = [m for m in match_data.get('home_recent_matches_detailed', []) if m['venue'] == 'home']
        if h_home_games:
            # 假設 score 是 "主-客"，比薩在主場就是 score[0]
            h_scored_val = sum(int(m['score'].split('-')[0]) for m in h_home_games)
            h_scored_avg = h_scored_val / len(h_home_games)
        else:
            h_scored_avg = match_data.get('home_goals_scored', 0) / 19.0

        # 2. 主隊防守 (Home Conceded)
        if h_home_games:
            # 比薩在主場失球就是 score[1]
            h_conceded_val = sum(int(m['score'].split('-')[1]) for m in h_home_games)
            h_conceded_avg = h_conceded_val / len(h_home_games)
        else:
            h_conceded_avg = match_data.get('home_goals_conceded', 0) / 19.0

        # 3. 客隊進攻 (Away Scored)
        a_away_games = [m for m in match_data.get('away_recent_matches_detailed', []) if m['venue'] == 'away']
        if a_away_games:
            # 客隊在客場，入球是 score[1]
            a_scored_val = sum(int(m['score'].split('-')[1]) for m in a_away_games)
            a_scored_avg = a_scored_val / len(a_away_games)
        else:
            a_scored_avg = match_data.get('away_goals_scored', 0) / 19.0

        # 4. 客隊防守 (Away Conceded)
        if a_away_games:
            # 客隊在客場，失球是 score[0]
            a_conceded_val = sum(int(m['score'].split('-')[0]) for m in a_away_games)
            a_conceded_avg = a_conceded_val / len(a_away_games)
        else:
            a_conceded_avg = match_data.get('away_goals_conceded', 0) / 19.0


        # =================================================================
        # 📦 數據打包
        # =================================================================
        judge_h_data = {
            'injury_penalty': h_inj_pen,
            'rank': int(home_ranking) if str(home_ranking).isdigit() else 99,
            'win_rate': calculate_win_rate_helper(match_data.get('home_recent_form', [])),
            'recent_form': match_data.get('home_recent_form', []), 
            'momentum': h_mom_val,
            'fatigue_days': h_fatigue['days'],
            'conceded_avg': h_conceded_avg, # 使用精準計算值
            'scored_avg': h_scored_avg      # 使用精準計算值 (0.5)
        }

        judge_a_data = {
            'injury_penalty': a_inj_pen,
            'rank': int(away_ranking) if str(away_ranking).isdigit() else 99,
            'win_rate': calculate_win_rate_helper(match_data.get('away_recent_form', [])),
            'recent_form': match_data.get('away_recent_form', []),
            'momentum': a_mom_val,
            'fatigue_days': a_fatigue['days'],
            'conceded_avg': a_conceded_avg,
            'scored_avg': a_scored_avg
        }



        judge_odds_data = {
            'rise_home': rise_home,
            'rise_away': rise_away
        }
        
        # 預計算泊松覆蓋率
        temp_h_exp = poisson_result['home_expected_goals'] if poisson_result else 1.5
        temp_a_exp = poisson_result['away_expected_goals'] if poisson_result else 1.0
        temp_prob_h = self.calculate_handicap_coverage(temp_h_exp, temp_a_exp, handicap_val)
        temp_prob_a = self.calculate_handicap_coverage(temp_a_exp, temp_h_exp, -handicap_val)
        mom_side_check = "home" if mom_diff_val > 0 else "away"
        
        # [V3.1 Update] 嘗試從日期中提取月份
        current_month = 5 # 默認季尾
        try:
            if 'date' in match_data:
                # 簡單解析 YYYY-MM-DD
                date_str = str(match_data['date'])
                if '-' in date_str:
                    parts = date_str.split('-')
                    if len(parts) >= 2: current_month = int(parts[1])
        except: pass

        # [V3.6.4 Fix] 提取剋星狀態
        is_nemesis_active = '明顯剋星' in h2h_deep.get('nemesis_relationship', '')

        # 🛠️ [V6.0 新增] 調用市場共振分析 (這行是用來計算的)
        market_analysis = MarketResonanceV6.analyze_market_forces(match_data, handicap_val)

        judge_env_data = {
            'mom_diff': mom_diff_val,
            'target_prob': temp_prob_h if mom_side_check == "home" else temp_prob_a,
            'target_vol': h_volatility if mom_side_check == "home" else a_volatility,
            'handicap': handicap_val,
            'match_type': match_type,
            'month': current_month,
            'nemesis': is_nemesis_active, # 👈 已補上逗號
            'h2h_backtest': h2h_backtest, # 👈 已補上逗號
            'market_resonance': market_analysis # 👈 [V6.0] 將計算結果傳入去
        }

        # 2. 開庭審判 (Deliberate)
        # -----------------------------------------------------------
        # [V3.2 Patch] 使用 SilentHunter 法官
        # -----------------------------------------------------------
        arbiter = FinalJudgeV37_Clean()  # 👈 記得改呢度！
        judge_h_corr, judge_a_corr, judge_logs, judge_strategy = arbiter.deliberate(
            judge_h_data, judge_a_data, judge_odds_data, judge_env_data
        )


        # 3. 執行判決 (Execute Verdict)
        home_correction += judge_h_corr
        away_correction += judge_a_corr
        correction_msg.extend(judge_logs)
        
        if judge_strategy: strategy_used = judge_strategy
            
        # 導出關鍵狀態供報告顯示
        is_panic_exemption_triggered = arbiter.flags['is_panic_exemption_triggered']
        veto_triggered = arbiter.flags['veto_triggered']
        veto_msg = arbiter.flags['veto_msg']

        # ═══════════════════════════════════════════════════════════════


        home_total_score += home_correction
        away_total_score += away_correction

        # Quantum Leap V2.7.1
        current_handicap = handicap_val
        h_exp = poisson_result['home_expected_goals'] if poisson_result else 1.5
        a_exp = poisson_result['away_expected_goals'] if poisson_result else 1.0
        prob_home_cover = self.calculate_handicap_coverage(h_exp, a_exp, current_handicap)
        prob_away_cover = self.calculate_handicap_coverage(a_exp, h_exp, -current_handicap)

        h_gf_sim = h_exp * 1.2; h_ga_sim = a_exp * 0.8
        a_gf_sim = a_exp * 1.2; a_ga_sim = h_exp * 0.8
        style_h_bonus, style_a_bonus = self.calculate_style_mismatch(h_gf_sim, h_ga_sim, a_gf_sim, a_ga_sim)
        style_msg = ""
        if style_h_bonus > 0: style_msg += f"⚔️ 主隊風格剋制(+{style_h_bonus}) "
        if style_a_bonus > 0: style_msg += f"⚔️ 客隊風格剋制(+{style_a_bonus}) "

        h_trend_score = self.calculate_mean_reversion(match_data.get('home_recent_form', []))
        a_trend_score = self.calculate_mean_reversion(match_data.get('away_recent_form', []))
        reversion_msg = []
        if h_trend_score != 0: reversion_msg.append(f"主隊回歸修正: {h_trend_score}")
        if a_trend_score != 0: reversion_msg.append(f"客隊回歸修正: {a_trend_score}")

        poisson_h_bonus = (prob_home_cover - 50) * 0.15 if prob_home_cover > 55 or prob_home_cover < 45 else 0
        poisson_a_bonus = (prob_away_cover - 50) * 0.15 if prob_away_cover > 55 or prob_away_cover < 45 else 0
        v271_home_adjust = style_h_bonus + h_trend_score + poisson_h_bonus
        v271_away_adjust = style_a_bonus + a_trend_score + poisson_a_bonus
        home_total_score += v271_home_adjust; away_total_score += v271_away_adjust
        
        # Analyst Vision V2.7.6
        h_momentum = AdvancedMetrics.calculate_weighted_momentum(h_recent_rev)
        a_momentum = AdvancedMetrics.calculate_weighted_momentum(a_recent_rev)
        mom_diff = h_momentum - a_momentum
        mom_correction = mom_diff * 0.25 
        home_total_score += mom_correction
        mom_msg = f"主{h_momentum:.0f} vs 客{a_momentum:.0f}"

        quarter_correction = 0.0; quarter_msg = ""
        is_quarter = (abs(handicap_val) * 4) % 2 != 0 
        if is_quarter:
            league_draw_rate = league_info.get('draw_rate', 0.27)
            if abs(handicap_val) == 0.25 and league_draw_rate > 0.28: 
                if handicap_val > 0: quarter_correction -= 8.0; quarter_msg = "⚖️ [半盤博弈] 主讓平半且平局率高，上盤高險"
                else: quarter_correction += 8.0; quarter_msg = "⚖️ [半盤博弈] 客讓平半且平局率高，上盤高險"
            home_total_score += quarter_correction

        # 執行判決
        home_total_score = max(10, min(99, home_total_score))
        away_total_score = max(10, min(99, away_total_score))
        
        if match_data.get('forced_risk_level') == '🔴 極高風險':
            home_total_score = 50.0; away_total_score = 50.0
        
        score_diff = home_total_score - away_total_score
        home_win_prob = max(0.1, min(0.9, 0.5 + (score_diff / 200)))
        home_kelly = (home_win_prob * home_odds - 1) / (home_odds - 1) if home_odds > 1 else 0
        away_kelly = ((1-home_win_prob) * away_odds - 1) / (away_odds - 1) if away_odds > 1 else 0
        
        if home_kelly > away_kelly:
            rec_side = 'home'; rec_team = match_data['home_team']; rec_kelly = max(0, home_kelly); rec_odds = home_odds
        else:
            rec_side = 'away'; rec_team = match_data['away_team']; rec_kelly = max(0, away_kelly); rec_odds = away_odds

        draw_risk_val = 0.28
        if match_data.get('forced_draw_risk_increase'): draw_risk_val = 0.45

        risk_eval = self.risk_evaluator.evaluate_comprehensive_risk(
            draw_risk=draw_risk_val, heavy_defeat_risk=poisson_result['heavy_defeat_risk'], 
            defense_level=home_collapse['level'], consistency='高度一致', 
            adaptation_score=home_hdp_perf['adaptation_score']
        )
        if match_data.get('forced_risk_level') == '🔴 極高風險':
            risk_eval['score'] = 99; risk_eval['level'] = '🔴 極高風險(鎖定)'

# =================================================================
        # 🛡️ V3.7 核心接駁：三權分立校驗 (Tri-Check Wiring)
        # =================================================================
        score_diff = home_total_score - away_total_score
            
        # 提取 Pin 變動
        pin_chg_check = 0.0
        if 'company_odds' in match_data and 'PIN' in match_data['company_odds']:
             p = match_data['company_odds']['PIN']
             if p.get('early_home') and p.get('current_home'):
                 pin_chg_check = p['current_home'] - p['early_home']
            
        # 呼叫校驗器
        # ==========================================================
        # 🛡️ [SYSTEM UPGRADE] 調用 V5.0 Ultimate Final (全能上帝版)
        # ==========================================================
        
        # 1. [數據適配] 為 V5.0 構建專屬數據包 (Data Packaging)
        # V5.0 需要結構化的 stats 字典，這裡我們將現有數據打包
        match_data['home_stats'] = {
            'home_win_rate': home_perf.get('win_rate', 0.33),
            'conceded_avg': match_data.get('home_goals_conceded', 0) / 5.0, # 假設基於15場平均
            'goals_scored': match_data.get('home_goals_scored', 0)
        }
        match_data['away_stats'] = {
            'away_win_rate': away_perf.get('win_rate', 0.11), 
            'conceded_avg': match_data.get('away_goals_conceded', 0) / 5.0,
            'goals_scored': match_data.get('away_goals_scored', 0)
        }
        
        # 傳入對手排名 (用於 V4.1 欺善怕惡)
        # 這裡簡單判斷：如果是主場，對手就是客隊排名
        match_data['opponent_rank'] = away_ranking

        # 2. [參數準備]
        base_score_diff = home_total_score - away_total_score
        odds_trend_data = {'pin_change': pin_chg_h} 
        
                # 3. [核心執行] V5.0 Ultimate 驗證
        # 注意：這裡傳入的參數名 (odds_trend_data, risk_eval) 請保持你原本代碼中的名稱
        v50_result = PrecisionValidatorV50_Ultimate.validate_decision(
            match_data, 
            base_score_diff, 
            odds_trend_data, 
            risk_eval['level']
        )
        
        # 🚨【關鍵修復】建立變數橋樑，防止後續代碼報錯
        v37_res = v50_result  # <--- 必須加上這一行！
        
        # 4. [結果提取 & 兼容性映射]
        # 關鍵：將 V5.0 的結果賦值給 v37 變數，這樣報告模板就能直接讀取，不用改模板
        v37_status = v50_result['status']
        v37_confidence = v50_result['confidence']
        v37_log = v50_result['log']

        
        # 5. [熔斷決策]
        if v50_status == "SKIP":
            force_no_recommend = True
            quality_eval['should_recommend'] = False
            quality_eval['reasons'].append(f"🛑 [V5.0 攔截] {v50_log}")
        elif "BET" in v50_status:
            # 如果 V5.0 強烈推薦 (例如 God Mode)，給予額外信心加成
            if v50_confidence > 0.9:
                quality_eval['confidence_level'] = "🔥 極高 (上帝模式)"
                correction_msg.append(f"👑 V5.0 上帝模式啟動: {v50_log}")

    
        # 覆寫決策
        quality_eval = self.smart_no_recommendation.evaluate_recommendation_quality(rec_kelly, draw_risk_val, '🟢 正常', rec_side)
            
        if v37_res['status'] == "SKIP":
            quality_eval['should_recommend'] = False
            quality_eval['reasons'].append(f"🛡️ [V3.7 觀望] {v37_res['log']}")
            optimal_bet = None
        else:
            quality_eval['should_recommend'] = True
            quality_eval['confidence_level'] = "極高" if v37_res['confidence'] > 0.8 else "中"
            optimal_bet = self.kelly_optimizer.calculate_optimal_bet(rec_kelly, self.bankroll, self.risk_preference, quality_eval['confidence_level'])
                
        final_reasoning = f"【V3.7 架構】\n🛡️ 校驗: {v37_res['log']}\n" + " | ".join(correction_msg + quality_eval['reasons'])

        
        return {
        # [V6.9.1] 導出攻防數據供報告顯示
            'scored_avg_h':    judge_h_data['scored_avg'],
            'conceded_avg_h': judge_h_data['conceded_avg'],
            'scored_avg_a': judge_a_data['scored_avg'],
            'conceded_avg_a': judge_a_data['conceded_avg'],

            'home_team': match_data['home_team'], 'away_team': match_data['away_team'],
            'league': league_info['name'], 'handicap_info': handicap_info,
            'handicap_display': f"{handicap_info['display']} (Pin:{pin_hdp_input})", 'match_type': match_type,
            'system_version': 'V5.0 Ultimate Final (God Mode)',
            'home_total_score': home_total_score, 'away_total_score': away_total_score,
            'home_expected_goals': poisson_result['home_expected_goals'] if poisson_result else 0,
            'away_expected_goals': poisson_result['away_expected_goals'] if poisson_result else 0,
            'draw_risk': draw_risk_val * 100, 'consistency': '高度一致',
            'recommended_team': rec_team, 'recommended_kelly': rec_kelly, 
            'recommended_odds': rec_odds, 'quality_evaluation': quality_eval, 
            'optimal_bet': optimal_bet, 'strategy_used': strategy_used,
            'reasoning': final_reasoning,
            
            # 👇 [關鍵修改] 請將左邊嘅 Key 改做 v37，右邊保留 v50 變數
            # 這樣報告生成器才能讀取到數據！
            'v37_status': v50_status,       
            'v37_confidence': f"{v50_confidence:.2f}",
            'v37_log': v50_log,

            # (如果你想保留 v50 標籤做後備都可以，但上面三行必須要有)
            'v50_status': v50_status,
            'v50_confidence': f"{v50_confidence:.2f}",
            'v50_log': v50_log,

            'home_patch_effect': f"{home_correction:+.1f}", 'away_patch_effect': f"{away_correction:+.1f}",
            'home_correction': home_correction, 'away_correction': away_correction,
            'patch_message': " | ".join(correction_msg) if correction_msg else "無特殊修正",
            'prob_home_cover': prob_home_cover, 'prob_away_cover': prob_away_cover,
            'style_h_bonus': style_h_bonus, 'style_a_bonus': style_a_bonus, 'style_msg': style_msg,
            'h_trend_score': h_trend_score, 'a_trend_score': a_trend_score, 'reversion_msg': " | ".join(reversion_msg),
            'h_momentum_score': f"{h_momentum:.1f}", 'a_momentum_score': f"{a_momentum:.1f}", 'momentum_msg': mom_msg,
            'quarter_handicap_msg': quarter_msg if quarter_msg else "非敏感盤口",
            'nemesis_relationship': h2h_deep['nemesis_relationship'],
            'comprehensive_risk_score': risk_eval['score'], 'comprehensive_risk_level': risk_eval['level'],
            'h_volatility': f"{h_volatility:.2f}", 'a_volatility': f"{a_volatility:.2f}",
            'h_time_decay_label': h_time_decay['label'], 'a_time_decay_label': a_time_decay['label'],
            'h_fatigue_days': h_fatigue['days'], 'a_fatigue_days': a_fatigue['days'],
            'odds_source_name': match_data.get('odds_source_name', 'Manual'),
            'odds_trend_description': match_data.get('odds_change', {}).get('trend_description', '平穩'),
            'pin_early': f"{pin_data.get('early_home', '-')} / {pin_data.get('early_away', '-')}" if pin_data.get('early_home') else '-',
            'pin_current': f"{pin_data.get('current_home', '-')} / {pin_data.get('current_away', '-')}" if pin_data.get('current_home') else '-', 
            'pin_change': f"{pin_chg_h:+.2f}" if pin_data.get('early_home') else '-',
            'b365_early': f"{b365_data.get('early_home', '-')} / {b365_data.get('early_away', '-')}" if b365_data.get('early_home') else '-',
            'b365_current': f"{b365_data.get('current_home', '-')} / {b365_data.get('current_away', '-')}" if b365_data.get('current_home') else '-',
            'b365_change': f"{b365_chg_h:+.2f}" if b365_data.get('early_home') else '-',
            'avg_current': match_data.get('all_companies_data', {}).get('AVG', {}).get('home_odds', '-'),
            'home_defense_level': home_collapse['level'], 'away_defense_level': away_collapse['level'],
            'home_advantage_level': '標準', 'home_bonus': home_bonus,
            'home_ranking': home_ranking, 'away_ranking': away_ranking,
            'home_ranking_score': home_base_score, 'away_ranking_score': away_base_score,
            'veto_triggered': veto_triggered,
            'veto_msg': veto_msg,
            'diff_value': f"{abs(pin_chg_h - b365_chg_h):.2f}" if is_divergent else "0.00",
            'is_anchor_ban_triggered': is_anchor_ban_triggered,
            'mom_diff': f"{mom_diff:.1f}",
            'force_no_recommend': force_no_recommend,
            'is_extreme_lock_triggered': force_no_recommend,
            'has_nemesis_exemption': has_nemesis_exemption,
            'nemesis_type': nemesis_type,
            'h2h_bt_rate': f"{h2h_backtest.get('backtest_win_rate', 0):.0%}" if h2h_backtest else "N/A",
            'h2h_bt_total': h2h_backtest.get('total', 0) if h2h_backtest else 0,
            'h2h_bt_msg': h2h_backtest.get('msg', '無回測數據') if h2h_backtest else "-",
            'home_recent_form': " ".join(match_data.get('home_recent_form', [])[:5]),
            'away_recent_form': " ".join(match_data.get('away_recent_form', [])[:5]),
            'home_poisson': f"{poisson_result['home_expected_goals']:.2f}" if poisson_result else "N/A",
            'away_poisson': f"{poisson_result['away_expected_goals']:.2f}" if poisson_result else "N/A",
            'poisson_coverage': f"{prob_home_cover:.0f}" if rec_side == 'home' else f"{prob_away_cover:.0f}",
            'pin_trend': "升水" if pin_chg_h > 0 else ("降水" if pin_chg_h < 0 else "平穩"),
            'b365_trend': "升水" if b365_chg_h > 0 else ("降水" if b365_chg_h < 0 else "平穩"),
            'pin_diff': f"{pin_chg_h:+.2f}",
            'b365_diff': f"{b365_chg_h:+.2f}",
            'divergence_status': "嚴重分歧" if is_divergent else "正常",
            'diff_val': f"{abs(pin_chg_h - b365_chg_h):.2f}",
            'strategy_name': strategy_used,
            'base_home': f"{home_base_score:.1f}",
            'base_away': f"{away_base_score:.1f}",
            'correction_1': correction_msg[0] if len(correction_msg) > 0 else "無修正",
            'correction_2': correction_msg[1] if len(correction_msg) > 1 else "",
            'final_home': f"{home_total_score:.1f}",
            'final_away': f"{away_total_score:.1f}",
            'recommendation': f"{rec_team} {handicap_info['display']}",
            'confidence_level': quality_eval['confidence_level'],
            'risk_level': risk_eval['level'],
            'home_rank': home_ranking,
            'away_rank': away_ranking,
            'rank_diff': home_ranking - away_ranking,
            'home_form': " ".join(match_data.get('home_recent_form', [])[:6]),
            'away_form': " ".join(match_data.get('away_recent_form', [])[:6]),
            'handicap': handicap_info['display'],
            'handicap_recommendation': f"{rec_team} {'讓 ' + str(abs(handicap_info['value'])) if (rec_side == 'home' and handicap_info['value'] > 0) or (rec_side == 'away' and handicap_info['value'] < 0) else '受讓 ' + str(abs(handicap_info['value']))}",
            'handicap_odds': f"{rec_odds:.2f}",
            'handicap_kelly': f"{rec_kelly*100:.1f}",
            'handicap_bet': f"{optimal_bet['adjusted_kelly_bet']:.0f}" if optimal_bet else "0",
            'handicap_decision_icon': "✅" if quality_eval['should_recommend'] else "🚫",
            'ou_recommendation': "暫無推薦", 'ou_odds': "-", 'ou_kelly': "0", 'ou_bet': "0", 'ou_decision_icon': "-"
        }



# ... (上面是 SmartBettingSystemV293 的結尾，不要動) ...

# ═══════════════════════════════════════════════════════════════
# 🛠️ [補丁 3] 手動數據注入器 (Manual Data Injector)
# ═══════════════════════════════════════════════════════════════
# ⚠️ 用戶請在此貼上你的手動數據 (系統會自動優先讀取這裡的資料)
MANUAL_INPUT_BLOCK = """


import re
from datetime import datetime

class DataInjector:
    """
    [V6.9.2 Final Fixed] Safe Mode Data Injector (AI Adapter Edition)
    已修復：冒號比分崩潰問題、增強日期提取、保留所有原有邏輯
    """

    @staticmethod
    def inject_manual_data(text_data: str, match_data: dict) -> dict:
        print("[System] Injecting Data (V6.8 Safe Mode)...")
        
        if not text_data: 
            return match_data

        # --- PART 1: Init & Clean ---
        # 預先淨化 (防止亂碼)
        clean_text = text_data.replace('：', ':').replace('(', ' ').replace(')', ' ')
        
        # 初始化容器
        if 'manual_1x2' not in match_data:
            match_data['manual_1x2'] = {'early': 0.0, 'current': 0.0}
        if 'manual_kelly' not in match_data:
            match_data['manual_kelly'] = {'early': 0.0, 'current': 0.0}
        if 'manual_ou' not in match_data:
            match_data['manual_ou'] = {
                'trend': 'Flat', 
                'early_over': 0.0, 'early_under': 0.0, 
                'current_over': 0.0, 'current_under': 0.0
            }

        # --- PART 2: Regex Extraction (數值提取) ---
        
        # 1. Handicap
        handicap_match = re.search(r"目標盤口(?:HKJC)?:\s*(?P<line>.+)", clean_text, re.IGNORECASE)
        if handicap_match:
            match_data['manual_handicap_line'] = handicap_match.group("line").strip()

        # 2. 1x2 (Pin)
        p1x2_match = re.search(r"Pin\s*1x2:.*?即\s*([\d\.]+)", clean_text, re.IGNORECASE)
        if p1x2_match:
            match_data['manual_1x2']['current'] = float(p1x2_match.group(1))

        # 3. Odds (Pin & 365)
        pattern = r":.*?初\s*([\d\.]+)\s*/\s*([\d\.]+).*?即\s*([\d\.]+)\s*/\s*([\d\.]+)"
        pin_match = re.search(r"Pin" + pattern, clean_text, re.IGNORECASE)
        b365_match = re.search(r"365" + pattern, clean_text, re.IGNORECASE)
        
        active = pin_match if pin_match else b365_match
        if active:
            e_h, e_a, c_h, c_a = map(float, active.groups())
            company = "PIN" if pin_match else "B365"
            if 'company_odds' not in match_data: match_data['company_odds'] = {}
            data = {'early_home': e_h, 'early_away': e_a, 'current_home': c_h, 'current_away': c_a}
            match_data['company_odds'][company] = data
            match_data['manual_odds_data'] = data

        # 4. Kelly
        kelly_match = re.search(r"凱利:.*?即\s*([\d\.]+)", clean_text, re.IGNORECASE)
        if kelly_match:
            match_data['manual_kelly']['current'] = float(kelly_match.group(1))

        # 5. Over/Under
        ou_match = re.search(r"大小(?:水|球)?:.*?初\s*([\d\.]+)\s*/\s*([\d\.]+).*?即\s*([\d\.]+)\s*/\s*([\d\.]+)", clean_text, re.IGNORECASE)
        if ou_match:
            oe_h, oe_a, oc_h, oc_a = map(float, ou_match.groups())
            if oe_h < 1.5: oe_h += 1.0
            if oc_h < 1.5: oc_h += 1.0
            trend = 'Flat'
            diff = oc_h - oe_h
            if diff <= -0.03: trend = 'OverDrop'
            elif diff >= 0.03: trend = 'UnderDrop'
            match_data['manual_ou'] = {'trend': trend, 'early_over': oe_h, 'early_under': oe_a, 'current_over': oc_h, 'current_under': oc_a}

        # [V6.9 New] 6. Goals Conceded (防火牆協議數據源)
        # 抓取格式： "主失球: 15" 或 "Home GA: 15"
        h_ga = re.search(r"(?:主|Home)\s*(?:失球|GA|Conceded)[:\s]*(\d+)", clean_text, re.IGNORECASE)
        a_ga = re.search(r"(?:客|Away)\s*(?:失球|GA|Conceded)[:\s]*(\d+)", clean_text, re.IGNORECASE)
        
        if h_ga: match_data['home_goals_conceded'] = int(h_ga.group(1))
        if a_ga: match_data['away_goals_conceded'] = int(a_ga.group(1))
        
    
        # [新增] 讀取進攻數據 (Goals For / GF)
        h_gf = re.search(r"(?:主|Home)\s*(?:入球|GF|Scored)[:\s]*(\d+)", clean_text, re.IGNORECASE)
        a_gf = re.search(r"(?:客|Away)\s*(?:入球|GF|Scored)[:\s]*(\d+)", clean_text, re.IGNORECASE)
        
        if h_gf: match_data['home_goals_scored'] = int(h_gf.group(1))
        if a_gf: match_data['away_goals_scored'] = int(a_gf.group(1))

        
        # 7. Injury & Date
        injury_match = re.search(r"傷停:\s*(.+)", clean_text)
        if injury_match:
            content = injury_match.group(1).strip()
            match_data['home_injury_text'] = content
            match_data['away_injury_text'] = content

        date_match = re.search(r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})|(\d{1,2}[-/]\d{1,2})', clean_text)
        if date_match and len(date_match.group(0)) > 3:
            match_data['date'] = date_match.group(0)

        # --- PART 3: Smart Scan (隊名與比分) ---
        
        raw_home = match_data.get('home_team', '').strip()
        raw_away = match_data.get('away_team', '').strip()

        # [Safe Mode] 物理分割法 (禁止 Regex 處理括號)
        t_home = raw_home.split('(')[0].split('（')[0].strip()
        t_away = raw_away.split('(')[0].split('（')[0].strip()

        if len(t_home) < 2: t_home = raw_home
        if len(t_away) < 2: t_away = raw_away

        h2h, home_rec, away_rec = [], [], []
        lines = text_data.strip().split('\n')
        curr_year = datetime.now().year

        for line in lines:
            line = line.strip()
            if not line: continue

            # 尋找比分
            score_match = re.search(r'\b(\d+)\s*[-:]\s*(\d+)\b', line)
            
            # 邏輯 1: 詳細比分
            if score_match and '.' not in score_match.group(0):
                # 🛠️ [CRITICAL FIX] 強制將冒號轉為橫線，防止 split('-') 報錯
                score = score_match.group(0).replace(':', '-')
                
                dm = re.search(r'\d{1,2}[-/]\d{1,2}', line)
                m_date = dm.group(0) if dm else f"{curr_year}-01-01"
                
                s_idx = line.find(score_match.group(0)) # 使用原始匹配字串找位置
                item = {'date': m_date, 'score': score}
                
                if t_home in line and t_away in line:
                    item['venue'] = 'home' if line.find(t_home) < s_idx else 'away'
                    h2h.append(item)
                elif t_home in line:
                    item['venue'] = 'home' if line.find(t_home) < s_idx else 'away'
                    home_rec.append(item)
                elif t_away in line:
                    item['venue'] = 'home' if line.find(t_away) < s_idx else 'away'
                    away_rec.append(item)

            # 邏輯 2: WDL 文字
            else:
                upper = line.upper()
                # 排除無關行
                if "PIN" in upper or "BET365" in upper or "HKJC" in upper: continue

                wdl = [c for c in upper if c in ['W', 'D', 'L']]
                if not wdl: continue
                
                if "對賽" in line or "H2H" in line:
                    if 'h2h_recent_form' not in match_data: match_data['h2h_recent_form'] = wdl
                elif "主" in line and ("近況" in line or "FORM" in upper):
                    if 'home_recent_form' not in match_data: match_data['home_recent_form'] = wdl
                elif "客" in line and ("近況" in line or "FORM" in upper):
                    if 'away_recent_form' not in match_data: match_data['away_recent_form'] = wdl

        # --- PART 4: Final Injection ---
        
        def to_form(details):
            f = []
            for m in details:
                try:
                    # 這裡依賴 score 必須是 "X-Y" 格式
                    h, a = map(int, m['score'].split('-'))
                    my = h if m['venue'] == 'home' else a
                    opp = a if m['venue'] == 'home' else h
                    if my > opp: f.append('W')
                    elif my == opp: f.append('D')
                    else: f.append('L')
                except: pass
            return f

        if h2h:
            match_data['h2h_details'] = h2h
            match_data['h2h_recent_form'] = to_form(h2h)

        if home_rec:
            match_data['home_recent_matches_detailed'] = home_rec
            match_data['home_recent_form'] = to_form(home_rec)

        if away_rec:
            match_data['away_recent_matches_detailed'] = away_rec
            match_data['away_recent_form'] = to_form(away_rec)
            
        return match_data


# ... (上面係 class SmartBettingSystemV293 嘅內容，唔好動) ...

# ═══════════════════════════════════════════════════════════════
# 📝 [V6.0 Final] 報告生成器 (Report Generator)
# ═══════════════════════════════════════════════════════════════
def generate_markdown_report(data: dict) -> str:
    """
    [V6.0] 支援凱利、大小球、歐亞錨定的完整報告生成器
    """
    # 1. 輔助函數 (避免數據缺失導致報錯)
    def g(key, default='-'): 
        val = data.get(key, default)
        return val if val is not None else default
        
    def f(key, fmt='{:.1f}'): 
        try: return fmt.format(float(data.get(key, 0)))
        except: return '0.0'
        
    # 🟢 [V6.9.2 修正] 數據同步協議 (Data Sync Protocol)
    # 我們不再這裡重新計算，而是直接信任主系統 (analyze_match) 傳來的 'scored_avg'
    # 這樣才能確保 "主場蟲" (0.5) 的計算結果被正確顯示在 Log 中
    
    # 防呆機制：萬一主系統無傳過來，先至用後備方案 (除以 5)
    if 'scored_avg_h' not in data: data['scored_avg_h'] = data.get('home_goals_scored', 0) / 5.0
    if 'conceded_avg_h' not in data: data['conceded_avg_h'] = data.get('home_goals_conceded', 0) / 5.0
    if 'scored_avg_a' not in data: data['scored_avg_a'] = data.get('away_goals_scored', 0) / 5.0
    if 'conceded_avg_a' not in data: data['conceded_avg_a'] = data.get('away_goals_conceded', 0) / 5.0


    # 2. 提取 V6.0 市場共振數據 (Market Resonance)
    mr = data.get('market_resonance', {})
    
    # --- 凱利數據格式化 ---
    kelly_early = mr.get('kelly_early', '-')
    kelly_curr = mr.get('kelly_curr', '-')
    kelly_sig = mr.get('kelly_signal', 'Neutral')
    
    # 設定凱利圖標
    if kelly_sig in ['Guard', 'SuperGuard']: kelly_icon = "🛡️"
    elif kelly_sig == 'Trap': kelly_icon = "🚨"
    else: kelly_icon = "-"
    
    # --- 大小球數據格式化 ---
    ou_trend = mr.get('ou_trend', 'Flat')
    ou_icon = "🌊" if ou_trend != 'Flat' else "-"
    
    # --- 歐亞錨定格式化 ---
    euro_odds = mr.get('euro_odds', 0)
    theo_hdp = mr.get('theo_hdp', 0)
    theo_diff = mr.get('theo_diff', 0)
    
    anchor_msg = "正常"
    if theo_diff >= 0.5: anchor_msg = "⚓ 歐亞陷阱 (誘盤)"
    elif theo_diff <= -0.5: anchor_msg = "🛡️ 莊家信心 (防範)"

    # 3. 計算波膽建議 (保持不變)
    try:
        h_exp = float(data.get('home_expected_goals', 1.0))
        a_exp = float(data.get('away_expected_goals', 1.0))
        h_i, a_i = int(round(h_exp)), int(round(a_exp))
        scores = set()
        scores.add(f"{h_i}-{a_i}")
        if h_exp > a_exp: 
            scores.add(f"{h_i+1}-{a_i}")
            scores.add(f"{h_i}-{max(0, a_i-1)}")
        else:
            scores.add(f"{h_i}-{a_i+1}")
            scores.add(f"{max(0, h_i-1)}-{a_i}")
        score_str = ", ".join(sorted(list(scores)))
    except: score_str = "N/A"

    # 4. 計算信心指數星星
    try:
        v37_conf = float(data.get('v37_confidence', 0))
        if v37_conf > 0.8: stars = "⭐⭐⭐⭐⭐ (極強)"
        elif v37_conf > 0.6: stars = "⭐⭐⭐⭐ (強)"
        elif v37_conf > 0.4: stars = "⭐⭐⭐ (中)"
        else: stars = "⭐⭐ (觀望)"
    except: stars = "⭐⭐"

    # 5. 賠率變動字串處理 (處理 "1.90 / 2.00" 這種格式)
    def get_chg(curr, early):
        try:
            c = float(str(curr).split('/')[0])
            e = float(str(early).split('/')[0])
            diff = c - e
            return f"{diff:+.2f}"
        except: return "-"

    pin_chg_str = get_chg(g('pin_current'), g('pin_early'))
    b365_chg_str = get_chg(g('b365_current'), g('b365_early'))

            # 6. 構建報告 (V6.7 格式)
        report = f"""
# ✨ 智能投注系統 V6.9.2 GOD MODE 分析報告

## 1. 賽事與數據源
- **比賽：** {g('home_team')} vs {g('away_team')}
- **聯賽：** {g('league')} ({g('match_type')})
- **策略模式：** {g('strategy_used')}
- **盤口：** {g('handicap_display')}
- **賠率基準：** {g('odds_source_name')}
系統版本：V6.9.2 IronLogic (AI Data Adapter / Firewall / Dead-Line)

## 2. 資金市場與賠率監控 (Market & Kelly)
| 項目 | 初盤 | 即時 | 變動/信號 |
| :--- | :---: | :---: | :---: |
| **Pinnacle** | {g('pin_early')} | {g('pin_current')} | {pin_chg_str} |
| **Bet365** | {g('b365_early')} | {g('b365_current')} | {b365_chg_str} |
| **主勝凱利** | {mr.get('kelly_early', '-')} | {kelly_curr} | **{kelly_icon} ({mr.get('kelly_signal')})** |
| **大小球水** | {mr.get('ou_early', '-')} | {mr.get('ou_current', '-')} | **{ou_icon} ({ou_trend})** |

## 3. 核心運算細節
| 項目 | {g('home_team')} (主) | {g('away_team')} (客) | 運算備註 |
| :--- | :---: | :---: | :--- |
| **排名/底蘊** | {g('home_ranking')} | {g('away_ranking')} | 評分: {f('home_ranking_score')} vs {f('away_ranking_score')} |
| **近期狀態** | {" ".join(g('home_recent_form', [])[:5])} | {" ".join(g('away_recent_form', [])[:5])} | ⚖️ {g('reversion_msg', '狀態正常')} |
| **主場/戰意** | {g('home_advantage_level')} | - | 加成: +{g('home_bonus')} |
| **⚓ 歐亞錨定** | 歐賠: {euro_odds} | 理論讓: {theo_hdp} | **Diff: {theo_diff:.2f} ({anchor_msg})** |
| **🛡️ 智能修正** | **{f('home_correction', '{:+.1f}')}** | **{f('away_correction', '{:+.1f}')}** | **{g('patch_message')}** |

## 4. V6.0 全息視界數據
| 指標 | {g('home_team')} | {g('away_team')} | 影響 |
| :--- | :---: | :---: | :--- |
| **🌊 動量評分** | **{g('h_momentum_score')}** | **{g('a_momentum_score')}** | **{g('momentum_msg')}** |
| **⚛️ 泊松覆蓋** | **主贏盤 {g('prob_home_cover'):.0f}%** | **客贏盤 {g('prob_away_cover'):.0f}%** | **{g('poisson_msg', '正常')}** |
| **🛡️ 風格配對** | **{f('style_h_bonus', '{:+.1f}')}** | **{f('style_a_bonus', '{:+.1f}')}** | **{g('style_msg')}** |
| **📈 均值回歸** | **{f('h_trend_score', '{:+.1f}')}** | **{f('a_trend_score', '{:+.1f}')}** | **✅ 情緒正常** |
| **📉 波動極性** | {g('h_volatility')} | {g('a_volatility')} | **✅ 正常** |
| **💉 疲勞/腎上腺** | 休 {g('h_fatigue_days')} 天 | 休 {g('a_fatigue_days')} 天 | **{g('fatigue_msg', '🟢 充沛')}** |
| **⚔️ 剋星關係** | - | - | **{g('nemesis_relationship')}** |

## 5. 預測與風險
- **泊松預期比分：** {g('home_poisson')} : {g('away_poisson')}
- **綜合評分：** 主 {f('final_home')} vs 客 {f('final_away')}
- **平局風險：** {f('draw_risk', '{:.0f}')}% | **模型一致性：** {g('consistency')}
- **🔮 波膽建議：** **{score_str}**
- **📊 信心指數：** **{stars}**
- **🛡️ 綜合風險分：** **{g('comprehensive_risk_score')}** / 100 ({g('comprehensive_risk_level')})

---
### 🎯 最終決策 (Final Decision)
| 玩法 | 推薦選項 | 參考賠率 (歐) | 凱利值 | 建議注碼 | 決策 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **讓球盤** | **{g('handicap_recommendation')}** | {g('handicap_odds')} | {g('handicap_kelly')}% | HK$ {g('handicap_bet')} | {g('handicap_decision_icon')} |
| **大小球** | **{g('ou_recommendation')}** | - | - | HK$ {g('ou_bet')} | {g('ou_decision_icon')} |

# 🛠️ [Step 10.X 關鍵數據提取協議]
1. **[V6.0 市場共振]** 凱利: {mr.get('kelly_signal')} | 大小趨勢: {ou_trend} | 歐亞Diff: {theo_diff:.2f}
2. **[V3.7 三權分立]** 狀態: {g('v37_status')} | 信心: {g('v37_confidence')} | 校驗: {g('v37_log')}
3. **[聰明錢與紅線]** 策略標籤: {g('strategy_used')}
4. **[市場敬畏]** 否決觸發: {g('veto_triggered')} | 原因: {g('veto_msg')}
5. **[V6.9 防火牆與修正]** 主修正: {f('home_correction', '{:+.1f}')} | 客修正: {f('away_correction', '{:+.1f}')}
6. **[V6.9.2 攻防數據]** 疲勞: {g('h_fatigue_days')}/{g('a_fatigue_days')} | **真實均得/均失: {f('scored_avg_h', '{:.2f}')}/{f('conceded_avg_h', '{:.2f}')} vs {f('scored_avg_a', '{:.2f}')}/{f('conceded_avg_a', '{:.2f}')}**
7. **[智能豁免]** 鎖定: {g('force_no_recommend')} | 豁免: {g('has_nemesis_exemption')}

**💡 綜合理由：**
{g('reasoning')}
"""

    return report


def main():
    st.title("⚽ 智能投注系統 V6.9.2 (iOS版)")

    # 側邊欄設定
    with st.sidebar:
        st.header("⚙️ 設定")
        bankroll = st.number_input("本金 ($)", value=10000, step=1000)

    # 介面分頁
    tab1, tab2 = st.tabs(["📸 影相/OCR", "📝 手動輸入"])

    ocr_result = ""

    with tab1:
        st.info("請上傳賠率圖或積分榜 (手機可直接影相)")
        uploaded_file = st.file_uploader("選擇圖片", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='預覽圖片', use_column_width=True)
            
            if st.button("開始識別文字 (OCR)"):
                with st.spinner('🔍 AI 正在讀取圖片文字...'):
                    try:
                        # 執行 OCR
                        text = pytesseract.image_to_string(image, lang='chi_tra+eng')
                        ocr_result = text
                        st.success("✅ 讀取成功！")
                    except Exception as e:
                        st.error(f"OCR 錯誤: {e}")

    # 數據確認區
    st.subheader("📊 數據確認區")
    raw_text = st.text_area(
        "請確認或修改數據:", 
        value=ocr_result if ocr_result else "",
        height=300,
        placeholder="圖片識別後的文字會出現在這裡，你也可以直接貼上文字..."
    )

    # 執行按鈕
    if st.button("🚀 啟動分析 (Analyze)", type="primary", use_container_width=True):
        if not raw_text:
            st.error("❌ 請先提供數據！")
        else:
            with st.spinner('🤖 V6.9.2 核心運算中...'):
                try:
                    # 1. 準備數據
                    match_data = {'raw_text': raw_text, 'bankroll': bankroll}
                    
                    # 2. 呼叫你的系統 (確保你上面已經貼咗 Class 代碼)
                    # 這裡會自動使用 DataInjector
                    if 'raw_text' in match_data:
                        match_data = DataInjector.inject_manual_data(match_data['raw_text'], match_data)
                    
                    system = SmartBettingSystemV293(bankroll=bankroll)
                    
                    # 3. 執行分析
                    report_data = system.analyze_match(match_data, ai_injury_feed=None)
                    
                    # 4. 生成報告
                    final_md = generate_markdown_report(report_data)
                    
                    st.markdown("---")
                    st.markdown(final_md)
                    
                except Exception as e:
                    st.error(f"❌ 運行錯誤: {str(e)}")
                    st.warning("請檢查你貼上的代碼是否完整 (Class DataInjector, SmartBettingSystemV293 等)")

if __name__ == "__main__":
    main()


