import math

class DriftAnalyzer:
    def __init__(self, p_value_threshold=0.05):
        self.p_value_threshold = p_value_threshold

    def calculate_basic_stats(self, base, prod):
        b_mean = sum(base) / len(base)
        p_mean = sum(prod) / len(prod)
        mean_shift = abs(p_mean - b_mean) / (abs(b_mean) + 1e-6)

        
        b_var = sum((x - b_mean)**2 for x in base) / max(len(base)-1, 1)
        p_var = sum((x - p_mean)**2 for x in prod) / max(len(prod)-1, 1)
        var_shift = abs(p_var - b_var) / (b_var + 1e-6)
        
        return mean_shift, var_shift

    def calculate_ks_statistic(self, base, prod):
        n1, n2 = len(base), len(prod)
        if n1 == 0 or n2 == 0: return 0.0, 1.0
        
        data_all = sorted(list(set(base + prod)))
        j1, j2 = 0, 0
        max_d = 0.0
        
        base_sorted = sorted(base)
        prod_sorted = sorted(prod)
        
        for val in data_all:
            while j1 < n1 and base_sorted[j1] <= val: j1 += 1
            while j2 < n2 and prod_sorted[j2] <= val: j2 += 1
            
            d = abs(j1/n1 - j2/n2)
            if d > max_d: max_d = d
            
        en = math.sqrt(n1 * n2 / (n1 + n2))
        try:
            p_val = 2.0 * math.exp(-2.0 * (en * max_d) ** 2)
        except OverflowError:
            p_val = 0.0
            
        return max_d, min(1.0, max(0.0, p_val))

    def _histogram(self, data, min_val, max_val, bins):
        counts = [0] * bins
        if min_val == max_val:
            counts[0] = len(data)
            return counts
            
        bin_width = (max_val - min_val) / bins
        for v in data:
            if v >= max_val:
                counts[-1] += 1
            else:
                idx = int((v - min_val) / bin_width)
                if 0 <= idx < bins:
                    counts[idx] += 1
        return counts

    def calculate_psi(self, base, prod, bins=10):
        if not base: return 0.0, "Stable"
        base_min, base_max = min(base), max(base)
        
        base_counts = self._histogram(base, base_min, base_max, bins)
        prod_counts = self._histogram(prod, base_min, base_max, bins)
        
        eps = 1e-4
        b_sum = sum(base_counts) + bins * eps
        p_sum = sum(prod_counts) + bins * eps
        
        psi_value = 0.0
        for b_c, p_c in zip(base_counts, prod_counts):
            b_pct = (b_c + eps) / b_sum
            p_pct = (p_c + eps) / p_sum
            psi_value += (p_pct - b_pct) * math.log(p_pct / b_pct)
            
        if psi_value < 0.1:
            severity = "Stable"
        elif psi_value <= 0.25:
            severity = "Moderate"
        else:
            severity = "Significant"
            
        return psi_value, severity

    def calculate_js_divergence(self, base, prod, bins=10):
        if not base or not prod: return 0.0
        base_min = min(min(base), min(prod))
        base_max = max(max(base), max(prod))
        
        base_counts = self._histogram(base, base_min, base_max, bins)
        prod_counts = self._histogram(prod, base_min, base_max, bins)
        
        b_sum, p_sum = sum(base_counts), sum(prod_counts)
        if b_sum == 0 or p_sum == 0: return 0.0
        
        js_div = 0.0
        for b_c, p_c in zip(base_counts, prod_counts):
            p = b_c / b_sum
            q = p_c / p_sum
            m = 0.5 * (p + q)
            
            if p > 0: js_div += 0.5 * p * math.log2(p / m)
            if q > 0: js_div += 0.5 * q * math.log2(q / m)
                
        return js_div

    def _parse_csv(self, csv_str):
        lines = csv_str.strip().split('\n')
        if not lines: return [], []
        headers = lines[0].strip().split(',')
        
        data = {h: [] for h in headers}
        
        for line in lines[1:]:
            parts = line.strip().split(',')
            for i, h in enumerate(headers):
                if i < len(parts):
                    data[h].append(parts[i])
                    
        return headers, data

    def detect_drift(self, baseline_str, prod_str):
        b_headers, b_data_dict = self._parse_csv(baseline_str)
        p_headers, p_data_dict = self._parse_csv(prod_str)
        
        drift_results = []
        
        for col in b_headers:
            if col.lower() in ("id", "customerid", "churn"):
                continue
                
            b_list, p_list = [], []
            try:
                for val in b_data_dict.get(col, []):
                    if val.strip(): b_list.append(float(val))
                for val in p_data_dict.get(col, []):
                    if val.strip(): p_list.append(float(val))
            except ValueError:
                continue
                
            if len(b_list) == 0 or len(p_list) == 0:
                continue
                
            stat, p_val = self.calculate_ks_statistic(b_list, p_list)
            psi_val, psi_severity = self.calculate_psi(b_list, p_list)
            js_div = self.calculate_js_divergence(b_list, p_list)
            mean_shft, var_shft = self.calculate_basic_stats(b_list, p_list)
            
            b_min = min(min(b_list), min(p_list))
            b_max = max(max(b_list), max(p_list))
            bins = 10
            base_counts = self._histogram(b_list, b_min, b_max, bins)
            prod_counts = self._histogram(p_list, b_min, b_max, bins)
            bin_labels = [round(b_min + i * (b_max - b_min)/bins, 2) for i in range(bins)]
            
            ks_score = 40.0 if p_val >= self.p_value_threshold else 0.0
            psi_penalty = min(psi_val / 0.25, 1.0)
            psi_score = 35.0 * (1 - psi_penalty)
            js_penalty = min(js_div / 0.1, 1.0)
            js_score = 25.0 * (1 - js_penalty)
            
            feature_health = ks_score + psi_score + js_score
            
            if p_val < self.p_value_threshold or psi_severity == "Significant" or js_div > 0.1 or mean_shft > 0.15:
                status = "Drift Detected"
            else:
                status = "No Drift"
            
            drift_results.append({
                "feature": col,
                "ks_statistic": round(stat, 4),
                "ks_p_value": round(p_val, 4),
                "psi_score": round(psi_val, 4),
                "psi_severity": psi_severity,
                "js_divergence": round(js_div, 4),
                "mean_shift": round(mean_shft, 4),
                "var_shift": round(var_shft, 4),
                "feature_health": round(feature_health, 2),
                "status": status,
                "base_counts": base_counts,
                "prod_counts": prod_counts,
                "bin_labels": bin_labels
            })
            
        total_features = len(drift_results)
        drifted_count = sum(1 for r in drift_results if r["status"] == "Drift Detected")
        
        avg_health = sum(r["feature_health"] for r in drift_results) / total_features if total_features > 0 else 100.0
            
        return {
            "summary": {
                "total_features": total_features,
                "drifted_features": drifted_count,
                "dataset_health_score": round(avg_health, 2),
                "methods_applied": ["KS (40%)", "PSI (35%)", "JS (25%)", "Mean/Var Shift"]
            },
            "feature_details": drift_results
        }
