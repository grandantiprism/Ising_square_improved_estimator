#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <string>
#include <chrono>
#include <queue>
#include <iomanip>

using namespace std;

// ==========================================
// シミュレーションパラメータの設定
// ==========================================
struct Config {
    const int L = 64;              // 格子の一辺の長さ
    const double beta_min = 0.42;  // 最小逆温度
    const double beta_max = 0.46;  // 最大逆温度
    const int beta_num = 20;       // betaの分割数
    
    const int n_warmup = 5000;     // 焼きなましステップ数
    const int n_measure = 1000000;   // 測定ステップ数
    
    const uint32_t seed = 12345;   // 乱数シード
};

class Ising2D {
private:
    int L, N;
    vector<int> spins;
    mt19937 engine;
    uniform_real_distribution<double> dist_real;
    uniform_int_distribution<int> dist_int;

public:
    Ising2D(int L, uint32_t seed) : L(L), N(L * L), spins(L * L, 1),
                                    engine(seed), dist_real(0.0, 1.0), dist_int(0, L * L - 1) {}

    inline int get_idx(int x, int y) {
        return ((x + L) % L) * L + ((y + L) % L);
    }

    int wolff_step(double beta) {
        int start_node = dist_int(engine);
        int old_spin = spins[start_node];
        int new_spin = -old_spin;

        double p_add = 1.0 - exp(-2.0 * beta);
        queue<int> q;

        spins[start_node] = new_spin;
        q.push(start_node);
        int cluster_size = 1;

        int dx[] = {1, -1, 0, 0};
        int dy[] = {0, 0, 1, -1};

        while (!q.empty()) {
            int curr = q.front();
            q.pop();

            int cx = curr / L;
            int cy = curr % L;

            for (int i = 0; i < 4; ++i) {
                int neighbor = get_idx(cx + dx[i], cy + dy[i]);
                if (spins[neighbor] == old_spin) {
                    if (dist_real(engine) < p_add) {
                        spins[neighbor] = new_spin;
                        q.push(neighbor);
                        cluster_size++;
                    }
                }
            }
        }
        return cluster_size;
    }

    // 全磁化 M = sum(s_i) / N を計算
    double calc_magnetization() {
        long long sum = 0;
        for (int s : spins) sum += s;
        return (double)sum / (L * L);
    }
};

int main() {
    Config conf;
    int total_sites = conf.L * conf.L;

    string res_filename = "ising_square_L" + to_string(conf.L) + ".csv";
    string log_filename = "sim_log.txt";

    ofstream ofs_res(res_filename);
    // CSVヘッダーに abs_m_naive を追加
    ofs_res << "beta,abs_m_naive,m2_improved,m4_naive,binder_cumulant" << endl;

    auto total_start = chrono::high_resolution_clock::now();
    
    Ising2D model(conf.L, conf.seed);
    cout << "Starting simulation: L=" << conf.L << "..." << endl;

    for (int i = 0; i < conf.beta_num + 1; ++i) {
        double beta = conf.beta_min + (conf.beta_max - conf.beta_min) * (double)i / conf.beta_num;
        
        // Warmup
        for (int s = 0; s < conf.n_warmup; ++s) model.wolff_step(beta);

        // Measurement
        double sum_abs_m = 0;  // |M| 用
        double sum_s = 0;      // <M^2> 改良推定量用
        double sum_m4 = 0;     // <M^4> 用
        
        for (int s = 0; s < conf.n_measure; ++s) {
            int s_size = model.wolff_step(beta);
            sum_s += (double)s_size;
            
            double m = model.calc_magnetization();
            sum_abs_m += abs(m);
            sum_m4 += pow(m, 4);
        }

        double abs_m = sum_abs_m / conf.n_measure;
        double m2_imp = sum_s / ((double)conf.n_measure * total_sites);
        double m4_naive = sum_m4 / conf.n_measure;
        double binder = m4_naive / (m2_imp * m2_imp);

        // CSV出力
        ofs_res << fixed << setprecision(8)
                << beta << ","
                << abs_m << ","
                << m2_imp << ","
                << m4_naive << ","
                << binder << endl;
        
        cout << "Finished beta=" << fixed << setprecision(4) << beta << endl;
    }
    
    auto total_end = chrono::high_resolution_clock::now();
    chrono::duration<double> total_elapsed = total_end - total_start;

    // ログファイルへの一括追記
    ofstream ofs_log(log_filename, ios::app);
    ofs_log << "L=" << conf.L
            << ", beta_min=" << conf.beta_min
            << ", beta_max=" << conf.beta_max
            << ", beta_num=" << conf.beta_num
            << ", n_measure=" << conf.n_measure
            << ", total_time=" << fixed << setprecision(2) << total_elapsed.count() << "s"
            << endl;

    cout << "\nResults saved: " << res_filename << endl;
    cout << "Total time: " << total_elapsed.count() << "s" << endl;

    return 0;
}
