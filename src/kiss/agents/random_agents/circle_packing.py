import numpy as np
import time
from scipy.optimize import minimize, linprog, dual_annealing
from scipy.sparse import lil_matrix, csc_matrix

def main(timeout, current_best_solution):
    """Circle packing: dual_annealing on centers + SLSQP polish."""
    n = 26
    start_time = time.time()

    best_score = -1.0
    best_circles = None

    if current_best_solution is not None:
        best_circles = current_best_solution.copy()
        best_score = float(np.sum(best_circles[:, 2]))

    # Precompute pair indices
    pair_i, pair_j = [], []
    for i in range(n):
        for j in range(i+1, n):
            pair_i.append(i)
            pair_j.append(j)
    pair_i = np.array(pair_i, dtype=int)
    pair_j = np.array(pair_j, dtype=int)
    num_pairs = len(pair_i)

    # LP setup (sparse)
    A_sp = lil_matrix((num_pairs, n))
    for k in range(num_pairs):
        A_sp[k, pair_i[k]] = 1.0
        A_sp[k, pair_j[k]] = 1.0
    A_sp = csc_matrix(A_sp)
    c_obj = -np.ones(n)

    def solve_lp(centers):
        wall = np.minimum(np.minimum(centers[:, 0], centers[:, 1]),
                          np.minimum(1 - centers[:, 0], 1 - centers[:, 1]))
        diff = centers[pair_i] - centers[pair_j]
        b_ub = np.sqrt(np.sum(diff ** 2, axis=1))
        bounds = [(0, float(wall[i])) for i in range(n)]
        res = linprog(c_obj, A_ub=A_sp, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return np.maximum(res.x, 0), -res.fun
        return None, 0.0

    # SLSQP setup
    num_cons = 4 * n + num_pairs
    idx_n = np.arange(n)

    def pack_x(centers, radii):
        x = np.zeros(3 * n)
        x[0::3] = centers[:, 0]
        x[1::3] = centers[:, 1]
        x[2::3] = radii
        return x

    def unpack_x(x):
        return np.column_stack([x[0::3], x[1::3]]), x[2::3].copy()

    def objective(x):
        return -np.sum(x[2::3])

    obj_jac = np.zeros(3 * n)
    obj_jac[2::3] = -1.0

    def objective_jac(x):
        return obj_jac

    def constraint_func(x):
        cons = np.zeros(num_cons)
        cx, cy, cr = x[0::3], x[1::3], x[2::3]
        cons[0:4*n:4] = cx - cr
        cons[1:4*n:4] = cy - cr
        cons[2:4*n:4] = 1.0 - cx - cr
        cons[3:4*n:4] = 1.0 - cy - cr
        dx = cx[pair_i] - cx[pair_j]
        dy = cy[pair_i] - cy[pair_j]
        cons[4*n:] = np.sqrt(dx*dx + dy*dy) - cr[pair_i] - cr[pair_j]
        return cons

    def constraint_jac(x):
        jac = np.zeros((num_cons, 3 * n))
        jac[4*idx_n, 3*idx_n] = 1.0;       jac[4*idx_n, 3*idx_n+2] = -1.0
        jac[4*idx_n+1, 3*idx_n+1] = 1.0;   jac[4*idx_n+1, 3*idx_n+2] = -1.0
        jac[4*idx_n+2, 3*idx_n] = -1.0;     jac[4*idx_n+2, 3*idx_n+2] = -1.0
        jac[4*idx_n+3, 3*idx_n+1] = -1.0;   jac[4*idx_n+3, 3*idx_n+2] = -1.0
        cx, cy = x[0::3], x[1::3]
        dx = cx[pair_i] - cx[pair_j]
        dy = cy[pair_i] - cy[pair_j]
        dists = np.maximum(np.sqrt(dx*dx + dy*dy), 1e-10)
        rows = 4*n + np.arange(num_pairs)
        ndx, ndy = dx / dists, dy / dists
        jac[rows, 3*pair_i] = ndx;     jac[rows, 3*pair_i+1] = ndy;   jac[rows, 3*pair_i+2] = -1.0
        jac[rows, 3*pair_j] = -ndx;    jac[rows, 3*pair_j+1] = -ndy;  jac[rows, 3*pair_j+2] = -1.0
        return jac

    slsqp_bounds = []
    for i in range(n):
        slsqp_bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.0, 0.5)])
    slsqp_constraints = {'type': 'ineq', 'fun': constraint_func, 'jac': constraint_jac}

    def validate_solution(centers, radii):
        if np.any(radii < -1e-6): return False
        if np.any(centers[:, 0] - radii < -1e-6): return False
        if np.any(centers[:, 1] - radii < -1e-6): return False
        if np.any(centers[:, 0] + radii > 1 + 1e-6): return False
        if np.any(centers[:, 1] + radii > 1 + 1e-6): return False
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < radii[i] + radii[j] - 1e-6:
                    return False
        return True

    def run_slsqp(centers, radii, maxiter=1000):
        x0 = pack_x(centers, radii)
        res = minimize(objective, x0, jac=objective_jac, method='SLSQP',
                      bounds=slsqp_bounds, constraints=slsqp_constraints,
                      options={'maxiter': maxiter, 'ftol': 1e-14})
        c, r = unpack_x(res.x)
        r = np.maximum(r, 0)
        if not validate_solution(c, r):
            fixed_r, fixed_s = solve_lp(c)
            if fixed_r is not None and validate_solution(c, fixed_r):
                return c, fixed_r, fixed_s
            return centers, radii, np.sum(radii)
        return c, r, np.sum(r)

    def update_best(centers, radii, score):
        nonlocal best_score, best_circles
        if score > best_score:
            best_score = score
            best_circles = np.hstack([centers, radii.reshape(-1, 1)])

    # --- Phase 1: Quick initial solution from hex grids ---
    configs = []
    for spacing in np.arange(0.10, 0.30, 0.005):
        configs.append(hex_grid_centers_v2(n, spacing))
    configs.append(hex_grid_centers(n))
    configs.append(grid_centers(n))
    for seed in range(10):
        configs.append(greedy_placement(n, seed))
    for rows in range(4, 7):
        for cols in range(4, 8):
            if rows * cols >= n:
                configs.append(rect_grid(n, rows, cols))

    scored_configs = []
    for centers in configs:
        radii, score = solve_lp(centers)
        if radii is not None:
            scored_configs.append((score, centers, radii))
    scored_configs.sort(key=lambda x: -x[0])

    for idx_cfg in range(min(5, len(scored_configs))):
        if time.time() - start_time > timeout * 0.10:
            break
        score, centers, radii = scored_configs[idx_cfg]
        sc, sr, ss = run_slsqp(centers, radii)
        update_best(sc, sr, ss)

    # --- Phase 2: dual_annealing on centers with LP objective ---
    eval_count = [0]
    da_best = [best_score]
    da_best_centers = [best_circles[:, :2].copy() if best_circles is not None else None]
    
    def center_obj(x_flat):
        eval_count[0] += 1
        centers = x_flat.reshape(n, 2)
        _, score = solve_lp(centers)
        if score > da_best[0]:
            da_best[0] = score
            da_best_centers[0] = centers.copy()
        return -score

    da_bounds = [(0.02, 0.98)] * (2 * n)
    
    # Use best known solution as initial point
    x0_da = best_circles[:, :2].flatten() if best_circles is not None else None
    
    # Run dual_annealing with time limit
    time_for_da = timeout * 0.65
    da_start = time.time()
    
    class TimeoutCallback:
        def __call__(self, x, f, context):
            if time.time() - da_start > time_for_da:
                return True
            return False
    
    try:
        da_result = dual_annealing(
            center_obj, da_bounds,
            x0=x0_da,
            maxiter=500,
            seed=42,
            callback=TimeoutCallback(),
            initial_temp=5230.0,
            restart_temp_ratio=2e-5,
            visit=2.62,
            accept=-5.0,
        )
    except Exception:
        pass
    
    # SLSQP polish the dual_annealing result
    if da_best_centers[0] is not None:
        da_c = da_best_centers[0]
        da_r, da_s = solve_lp(da_c)
        if da_r is not None:
            sc, sr, ss = run_slsqp(da_c, da_r, maxiter=2000)
            update_best(sc, sr, ss)

    # --- Phase 3: Perturbation + SLSQP from best ---
    if best_circles is not None:
        round_num = 0
        while time.time() - start_time < timeout * 0.92:
            bc = best_circles[:, :2].copy()
            rng = np.random.default_rng(round_num * 7 + 13)
            
            for scale in [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12]:
                if time.time() - start_time > timeout * 0.92:
                    break
                lp_candidates = []
                for seed in range(40):
                    rng2 = np.random.default_rng(seed + int(scale * 10000) + round_num * 100000)
                    perturbed = bc + rng2.normal(0, scale, size=bc.shape)
                    perturbed = np.clip(perturbed, 0.01, 0.99)
                    pr, ps = solve_lp(perturbed)
                    if pr is not None:
                        lp_candidates.append((ps, perturbed, pr))
                lp_candidates.sort(key=lambda x: -x[0])
                for _, cand_centers, cand_radii in lp_candidates[:3]:
                    if time.time() - start_time > timeout * 0.92:
                        break
                    sc, sr, ss = run_slsqp(cand_centers, cand_radii)
                    update_best(sc, sr, ss)
            round_num += 1

    # Final polish
    if best_circles is not None:
        sc, sr, ss = run_slsqp(best_circles[:, :2], best_circles[:, 2], maxiter=3000)
        update_best(sc, sr, ss)

    all_scores = [best_score]
    return {'circles': best_circles, 'all_scores': all_scores}


def greedy_placement(n, seed=0):
    rng = np.random.default_rng(seed)
    centers, radii = [], []
    c = np.clip(np.array([0.5 + rng.normal(0, 0.05), 0.5 + rng.normal(0, 0.05)]), 0.01, 0.99)
    centers.append(c)
    radii.append(min(c[0], c[1], 1-c[0], 1-c[1]))
    for _ in range(1, n):
        best_r, best_c = -1, None
        for _ in range(300):
            c = rng.uniform(0.01, 0.99, size=2)
            min_r = min(c[0], c[1], 1-c[0], 1-c[1])
            for j in range(len(centers)):
                min_r = min(min_r, np.linalg.norm(c - centers[j]) - radii[j])
            if min_r > best_r:
                best_r, best_c = min_r, c.copy()
        centers.append(best_c)
        radii.append(max(best_r, 0.001))
    return np.array(centers)

def rect_grid(n, rows, cols):
    centers = []
    dx, dy = 1.0 / (cols + 1), 1.0 / (rows + 1)
    for r in range(rows):
        for c in range(cols):
            if len(centers) >= n: break
            centers.append([(c + 1) * dx, (r + 1) * dy])
        if len(centers) >= n: break
    return np.array(centers[:n])

def hex_grid_centers(n):
    cols = int(np.ceil(np.sqrt(n * 2 / np.sqrt(3))))
    rows = int(np.ceil(n / cols))
    centers = []
    dx, dy = 1.0 / (cols + 1), 1.0 / (rows + 1)
    for row in range(rows):
        for col in range(cols):
            if len(centers) >= n: break
            x = (col + 1) * dx + (dx / 2 if row % 2 == 1 else 0)
            centers.append([x, (row + 1) * dy])
        if len(centers) >= n: break
    return np.clip(np.array(centers[:n]), 0.01, 0.99)

def hex_grid_centers_v2(n, spacing):
    centers = []
    y, row = spacing, 0
    while y < 1.0 - spacing / 2 and len(centers) < n * 2:
        x = spacing if row % 2 == 0 else spacing + spacing * 0.5
        while x < 1.0 - spacing / 2 and len(centers) < n * 2:
            centers.append([x, y])
            x += spacing
        y += spacing * np.sqrt(3) / 2
        row += 1
    if len(centers) > n:
        centers = np.array(centers)
        d = np.sqrt(np.sum((centers - 0.5) ** 2, axis=1))
        centers = centers[np.argsort(d)[:n]]
    elif len(centers) < n:
        centers = np.array(centers)
        extra = np.random.default_rng(0).uniform(0.05, 0.95, size=(n - len(centers), 2))
        centers = np.vstack([centers, extra])
    else:
        centers = np.array(centers)
    return np.clip(centers[:n], 0.01, 0.99)

def grid_centers(n):
    side = int(np.ceil(np.sqrt(n)))
    centers = []
    for i in range(side):
        for j in range(side):
            if len(centers) >= n: break
            centers.append([(i + 0.5) / side, (j + 0.5) / side])
        if len(centers) >= n: break
    return np.array(centers[:n])

if __name__ == "__main__":
    t0 = time.time()
    result = main(timeout=30, current_best_solution=None)
    elapsed = time.time() - t0
    score = sum(result['circles'][:, 2])
    print(f"Score (sum of radii): {score}")
    print(f"Negative of score: {-score}")
    print(f"Elapsed: {elapsed:.2f}s")
