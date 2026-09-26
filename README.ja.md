# Open MPI と OpenMP による並列素数探索

[English](README.md) | 日本語

C による分散メモリ(MPI)およびハイブリッド(MPI + OpenMP)の素数探索です。ベンチマーク
ハーネス、正当性チェック、2 ノードクラスタ用の SLURM ジョブ、そして計測した高速化率に対する
アムダールの法則に基づく分析を含みます。

Haruto Iriyama が 2 人チームの大学プロジェクトとして制作しました。比較対象となる逐次・
POSIX Threads・OpenMP のベースラインは
[parallel-primes-pthreads-openmp](https://github.com/Haruto03/parallel-primes-pthreads-openmp)
にあります。

## 内容

- **`task1.c`** — Open MPI による素数探索。4 種類の仕事分配戦略(ブロック、循環、
  ブロック循環、重み付きブロック)、2 種類のカーネル(試し割りと分割ふるい)、および
  ルートランクが報告するフェーズ別の時間内訳(broadcast / compute / gather / merge / write)を
  備えます。
- **`task2.c`** — 同じプログラムの Open MPI + OpenMP ハイブリッド版。各ランク内部で
  OpenMP のスケジュール指定(static / dynamic / guided)を選べます。
- **`verify.sh`** — すべての設定の出力を逐次ベースラインの出力と差分比較します。
- **`bench/`** — `run_bench.sh` が n・プロセス数・スレッド数・分配方法を掃引し、マシンごとに
  1 つの CSV を出力します。`plot.py` / `plot_caas.py` が高速化率・負荷分散・アムダールの
  グラフを `bench/graphs/` に描画します。8 スレッドのラップトップおよび 1 ノード / 2 ノード
  クラスタでの生の結果は `bench/results/` にあります。
- **`caas/`** — 大学の 2 ノードクラスタで実行した SLURM ジョブスクリプトと出力ログ
  (最大 32 ランク / 2 ノード、ノード間はギガビットイーサネット)。

## 結果の概要

![プロセス数に対する高速化率](bench/graphs/g3_speedup_vs_procs.png)
![実測値とアムダール予測の比較](bench/graphs/g6_mpi_empirical_vs_theory.png)

15 枚すべてのグラフは [`bench/graphs/`](bench/graphs/) にあります。高速化率は
(逐次的なファイル書き出しを含む)*プロセス全体*に対して計測しています。この書き出しこそが
達成可能な高速化率を制限する要因です。詳しくは下のアムダールの節を参照してください。

---

## ディレクトリ構成

```
parallel-primes-mpi/
├── task1.c            Open MPI による素数探索
├── task2.c            Open MPI + OpenMP ハイブリッドの素数探索
├── Makefile           mpicc -O2 -Wall -Wextra
├── verify.sh          正当性検証: 全設定 vs 逐次ベースラインの出力
├── bench/
│   ├── run_bench.sh   ベンチマークの掃引 -> results/<site>.csv(統一スキーマ)
│   ├── plot.py        CSV から高速化率・負荷分散・アムダールのグラフを描画
│   ├── plot_caas.py   CAAS の CSV から 1 ノード vs 2 ノードのグラフを描画
│   └── serial_sieve.c ふるいカーネル用の逐次ベースライン
└── caas/*.job         CAAS 用 SLURM ジョブ(run_bench.sh の薄いラッパー)
```

ベースラインのプログラムは
[parallel-primes-pthreads-openmp](https://github.com/Haruto03/parallel-primes-pthreads-openmp)
から取得しています(逐次 = task1.c、POSIX Threads = task2.c、OpenMP = task3.c)。これらの
プログラム自身のタイマーはファイル書き出しの前に止まるため、ハーネス側ではプロセス全体の
実時間を `total`、プログラムが報告する時間を `comp_max` として記録しています。

## 使い方

```
mpirun -np 8 ./task1 <n> [output_file] [dist] [chunk] [kernel]
mpirun -np 4 ./task2 <n> [output_file] [threads] [dist] [chunk] [sched] [sched_chunk] [kernel]

  dist   blockcyclic(既定) | block | cyclic | wblock
  sched  static | dynamic(既定) | guided        (OpenMP、task2 のみ)
  kernel trial(既定) | sieve                     (sieve は block/wblock/blockcyclic のみ対応)
```

どちらもフェーズ別の内訳(broadcast / compute / gather / merge / write)と、機械可読な
`CSV,...` 行を出力します。`PRIMES_VERBOSE=1` を指定すると、負荷分散の分析用にランクごとの
`RANK,...` 行が追加されます。

## ローカルでの作業手順(Docker)

```bash
docker run --rm -it -v "$(pwd)/..:/work" -w /work/parallel-primes-mpi <image-with-openmpi-and-gcc>
make && ./verify.sh                       # "ALL MATCH" が出れば成功
SITE=ryzen7535hs bench/run_bench.sh all   # 20〜30 分。結果は bench/results/ryzen7535hs.csv
python bench/plot.py bench/results/ryzen7535hs.csv --out bench/graphs --cores 8 --physical-cores 4   # Windows では matplotlib が必要
```

`run_bench.sh quick` を使うと 1 分以内にハーネスの動作確認ができます。`SITE` には意味のある
名前を設定してください(既定はコンテナのランダムなホスト名です)。1 台のマシンにつき 1 つの
ラベルを使い、複数のマシンのデータを 1 枚のグラフに混ぜないようにします。

ローカル環境で計測値を静かに壊す要因:

* `mpirun` には `--use-hwthread-cpus --bind-to none` が必要です(ハーネスに組み込み済み)。
  `--bind-to none` がないと Open MPI は np<=2 の実行を 1 コアに固定してしまい、
  1 × T のハイブリッド実行が実際より約 2.5 倍遅く見えます。
* 素数の出力ファイルはコンテナ内の `/tmp` に書いてください(ハーネスの既定)。バインド
  マウントした `/work` に書くと、Windows のマウント経由となり逐次の書き出しフェーズが
  実際より 3〜4 倍遅くなります。

## クラスタ(SLURM)での作業手順

```bash
scp -r parallel-primes-mpi parallel-primes-pthreads-openmp <user>@<cluster-headnode>:~/parallel-primes/
ssh <user>@<cluster-headnode>
cd ~/parallel-primes/parallel-primes-mpi/caas
sbatch serial_baseline.job      # 以降は 1 つずつ(1 ユーザーにつき実行中ジョブは 1 つ):
sbatch task1_np8_1node.job      # task1_np8_2node, task1_np16_1node, task1_np16_2node, task1_np32_2node
sbatch task2_np4x4_2node.job
sbatch evidence_np8_2node.job   # 素数ファイルを $HOME に書き出し、task1 と task2 の一致を確認
squeue -u $USER
```

各ジョブはローカル実行と同じスキーマで `../bench/results/caas-1node.csv` または
`caas-2node.csv` に追記するため、同じ `plot.py` がそのまま使えます:
`python bench/plot.py bench/results/caas-2node.csv --site caas-2node`。
`caas-1node`(共有メモリ転送)と `caas-2node`(ノード間ギガビットイーサネット)の比較が
`s_fabric` の実験にあたります。

## アムダール分析 — 各割合の測り方

すべての実行で、ルートランクの `MPI_Wtime` から次を報告します。

| フェーズ | アムダール上の位置づけ | 理由 |
|-----------|-------------|-----------------------------------------------------------|
| broadcast | 逐次 | 集団通信 1 回。コストは P とともに増加する |
| compute | 並列 | 各ランクのローカル素数探索の最大値 |
| gather | 逐次(+待ち) | Gather/Gatherv。最も遅いランクの待ち時間を含む |
| merge | 逐次 | ルートのみで行う k-way マージ(block/wblock ではスキップ) |
| write | 逐次 | ルートのみで行うファイル出力 |

`plot.py` は P = 1 の実行から `s = (bcast+gather+merge+write)/total` を求め、
`S(P) = 1/(s + (1-s)/P)` を実測の高速化率と並べて描画します(グラフ 6 / 7)。さらに、
各 P で実測した逐次部分を用いた変種(通信コストの増加が見えます)と、グスタフソンの曲線も
描画します。
