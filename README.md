# Break Blocks AI Training

ブロック崩しゲームの強化学習エージェントを訓練するためのプロジェクト。

## 環境セットアップ

### 依存関係のインストール

```bash
# CPU版PyTorch（推奨 - メモリ効率が良い）
uv sync

# GPU版PyTorch（RTX 4070等）を使う場合はpyproject.tomlを編集
# pytorch-cpu → pytorch-cu124 に変更後
uv sync
```

### PyTorch CPU vs CUDA

| 項目 | CPU版 | CUDA版 |
|------|-------|--------|
| メモリ使用量 | 低い | 高い（DLLロードで数GB消費） |
| 並列環境数 | 32-64環境可能 | 8-16環境が限界 |
| 推論速度 | 小バッチでは同等 | 大バッチで有利 |
| 推奨用途 | 本プロジェクト | 画像ベースRL |

**重要**: SubprocVecEnvを使用する場合、各サブプロセスでPyTorchがロードされます。CUDA版は各プロセスで巨大なDLLをロードするため、メモリエラーが発生しやすくなります。

## トレーニング

### 基本コマンド

```bash
# CPU版で32並列環境
uv run python train.py --device cpu --envs 32

# オプション
uv run python train.py --device cpu --envs 64 --timesteps 10000000
```

### 環境数とパフォーマンス

| 環境数 | 推定メモリ | 推定速度 | 10Mステップ所要時間 |
|--------|-----------|----------|---------------------|
| 8      | ~2GB      | ~700 steps/sec | ~4時間 |
| 16     | ~3GB      | ~1,200 steps/sec | ~2.5時間 |
| 32     | ~5GB      | ~1,800 steps/sec | ~1.5時間 |
| 64     | ~8GB      | ~2,500 steps/sec | ~1時間 |

## 観測空間（Observation Space）

216次元のベクトル:

| 範囲 | 内容 | 次元数 |
|------|------|--------|
| 0-1 | パドル位置 (x, width) | 2 |
| 2-5 | メインボール (x, y, vx, vy) | 4 |
| 6-25 | 追加ボール (5個 x 4) | 20 |
| 26-185 | ブロック状態 (8行 x 20列) | 160 |
| 186-209 | パワーアップ (8個 x 3: x, y, type) | 24 |
| 210 | 残りライフ | 1 |
| 211 | 残り時間 | 1 |
| 212 | 現在のコンボ | 1 |
| 213 | 最も近いパワーアップのY座標 | 1 |
| 214 | 最も近いパワーアップのタイプ | 1 |
| 215 | 最も近いパワーアップのX座標 | 1 |

**Tips**: パワーアップのX座標（215番目）を追加することで、アイテム収集行動が改善しました（勝率5%→10%）。

## 報酬設計

### 現在の報酬構造

```python
# 正の報酬
パドルヒット: +5.0        # ボールを維持する行動を強化
ブロック破壊: +20.0       # 各ブロック
コンボボーナス: +0.5 * combo
ステージクリア: +500.0    # 主目標
時間ボーナス: +200.0 * (残り時間/制限時間)

# パワーアップ収集
TIME_EXTEND: +20.0
PENETRATE: +15.0
MULTI_BALL: +10.0
SPEED_DOWN: +8.0
SPEED_UP: +3.0

# ペナルティ
ライフ喪失: -100.0
ゲームオーバー: -500.0
時間ペナルティ: -0.05/フレーム
```

### 報酬設計のポイント

1. **パドルヒット報酬が重要**: ボールを落とさない行動を学習させる
2. **ステージクリアを最大報酬に**: 主目標を明確化
3. **時間ペナルティは控えめに**: 大きすぎると早期終了を学習してしまう
4. **ゲームオーバーペナルティ > ライフ喪失ペナルティ**: 最後のライフを大切にする

## ハイパーパラメータ

### PPO設定（100%勝率を目指す場合）

```python
ppo_params = {
    'learning_rate': 1e-4,    # 低めで安定性重視
    'n_steps': 4096,          # 多めのステップで更新
    'batch_size': 128,        # 大きめのバッチ
    'n_epochs': 10,
    'gamma': 0.995,           # 高めで長期報酬重視
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.005,        # 低めで exploitation 重視
}

policy_kwargs = {
    'net_arch': [512, 512, 256]  # 大きめのネットワーク
}
```

### カリキュラム学習

初期段階ではカリキュラムを無効化し、ステージ1に集中することを推奨:

```python
curriculum = {
    'initial_max_stage': 1,
    'phase_steps': [],        # 空 = カリキュラム無効
    'time_penalty_scales': [1.0]
}
```

## 早期停止

95%勝率に達した時点で自動停止:

```python
WinRateEarlyStopCallback(
    eval_env=eval_env,
    target_win_rate=0.95,
    eval_freq=100000,         # 10万ステップごとに評価
    n_eval_episodes=50,       # 50エピソードで評価
    min_timesteps=500000,     # 最低50万ステップは学習
)
```

## プロファイリング結果

### ボトルネック分析

```
環境ステップ: 37.4% の時間
モデル推論:   62.6% の時間  ← ボトルネック
```

### スレッドスケーリング

```
1 thread:  3125 inferences/sec  ← 最速
2 threads: 2500 inferences/sec
4 threads: 2000 inferences/sec
8 threads: 1666 inferences/sec
```

**Tips**: 小さなバッチサイズでは、シングルスレッドが最速。`torch.set_num_threads(1)` を設定。

### バッチサイズスケーリング

```
Batch   8:  51,200 samples/sec
Batch  64: 170,666 samples/sec
Batch 128: 273,066 samples/sec  ← 5.3x speedup
```

### GPU vs CPU（推論）

| バッチサイズ | CPU | GPU | 勝者 |
|-------------|-----|-----|------|
| 8           | 3,333 | 2,777 | CPU |
| 64          | 5,000 | 5,000 | 同等 |
| 256         | 4,166 | 10,000 | GPU |
| 1024        | 4,166 | 16,666 | GPU |

**結論**: バッチサイズ256以上でGPUが有利。本プロジェクトではCPU推奨。

## モデルのエクスポート

### ONNX形式へのエクスポート

```bash
uv run python export_onnx.py
```

### ブラウザでの使用

`js/ai/RLAgent.js` でONNXモデルを読み込み、ブラウザ上で推論を実行。

## トラブルシューティング

### メモリエラー (WinError 1455)

```
OSError: [WinError 1455] ページング ファイルが小さすぎるため...
```

**原因**: CUDA版PyTorchのDLLがサブプロセスごとにロードされる

**解決策**:
1. CPU版PyTorchに切り替え
2. 環境数を減らす（32以下）
3. Windowsのページングファイルを増やす

### 勝率が上がらない

1. **パドルヒット報酬を追加**: ボール維持行動を学習させる
2. **観測空間を確認**: パワーアップのX座標が含まれているか
3. **カリキュラムを無効化**: まずステージ1だけに集中
4. **学習率を下げる**: 1e-4 程度
5. **過学習をチェック**: ベストモデルと最終モデルを比較

### ベストモデル vs 最終モデル

学習後半で性能が低下することがある（過学習）。`models/best_model/` に保存されたモデルを使用することを推奨。

```bash
uv run python evaluate.py models/best_model/best_model.zip
```

## ファイル構成と機能説明

### ルートディレクトリ（Pythonスクリプト）

| ファイル | 説明 |
|---------|------|
| `train.py` | **トレーニングエントリーポイント**。コマンドライン引数を受け取り `ai_training.training.train` を呼び出す |
| `evaluate.py` | **モデル評価スクリプト**。学習済みモデルの勝率・報酬・破壊ブロック数を測定 |
| `export_onnx.py` | **ONNXエクスポート**。学習済みモデルをブラウザ用ONNXフォーマットに変換 |
| `profile_training.py` | **ボトルネック分析**。環境ステップ vs モデル推論の時間配分を測定 |
| `profile_cpu.py` | **CPU/GPU比較**。スレッド数、バッチサイズ、デバイス別のパフォーマンス測定 |
| `plot_training.py` | **学習曲線可視化**。TensorBoardログから報酬・勝率グラフを生成 |
| `main.py` | 簡易実行用エントリーポイント |

---

### ai_training/env/ （ゲーム環境）

| ファイル | 説明 |
|---------|------|
| `breakout_env.py` | **Gymnasium環境クラス**。`BreakoutEnv`（基本環境）と`CurriculumBreakoutEnv`（カリキュラム学習対応）を定義。観測空間(216次元)、行動空間(3: 左/静止/右)、報酬計算を実装 |
| `game_simulation.py` | **ゲームシミュレーション**。JSゲームロジックをPythonに移植。ボール物理、パドル操作、衝突判定、パワーアップ処理を実装 |
| `constants.py` | **定数定義**。キャンバスサイズ、パドル/ボール/ブロック設定、ステージデータ(全10ステージ)、パワーアップ設定、観測次元数(216)を定義 |

#### breakout_env.py の主要クラス

```python
class BreakoutEnv(gym.Env):
    """基本環境"""
    def reset()     # 環境リセット、観測を返す
    def step(action) # 行動実行、(obs, reward, done, truncated, info)を返す
    def _calculate_reward(events) # 報酬計算（パドルヒット、ブロック破壊等）
    def _get_obs()  # 216次元の観測ベクトル生成

class CurriculumBreakoutEnv(BreakoutEnv):
    """カリキュラム学習対応環境"""
    def set_phase(phase) # 難易度フェーズを設定
    def _unlock_next_stage() # 勝率に応じてステージ解放
```

#### game_simulation.py の主要クラス

```python
class Ball:       # ボール物理（位置、速度、反射、貫通）
class Paddle:     # パドル操作（左右移動、当たり判定）
class Block:      # ブロック（HP、スコア、破壊判定）
class PowerUp:    # パワーアップアイテム（落下、収集）
class GameSimulation:  # ゲーム全体の管理
    def step(action, dt) # 1フレーム進行、イベント辞書を返す
    def get_observation() # 216次元ベクトル生成
```

---

### ai_training/training/ （学習関連）

| ファイル | 説明 |
|---------|------|
| `train.py` | **メイン学習スクリプト**。PPOモデル作成、環境生成、コールバック設定、学習ループを実装 |
| `callbacks.py` | **学習コールバック**。進捗ログ、チェックポイント保存、カリキュラム進行、早期停止を実装 |
| `reward.py` | **報酬計算ユーティリティ**。`RewardCalculator`クラスで報酬重みを設定可能 |

#### callbacks.py の主要クラス

```python
class TrainingCallback:
    """学習進捗モニタリング"""
    - 定期的に勝率・報酬をログ出力
    - チェックポイント自動保存

class CurriculumCallback:
    """カリキュラム学習制御"""
    - ステップ数に応じてフェーズ進行
    - 環境の難易度を動的に調整

class BestModelCallback:
    """ベストモデル保存"""
    - 評価環境で定期評価
    - 最高性能モデルを保存

class WinRateEarlyStopCallback:
    """早期停止"""
    - 目標勝率(95%)達成で学習終了
    - 最低ステップ数(50万)後から有効
```

---

### ai_training/export/ （モデルエクスポート）

| ファイル | 説明 |
|---------|------|
| `to_onnx.py` | **ONNXエクスポート**。PPOモデルからポリシーネットワークを抽出し、ONNX形式で保存 |
| `validate.py` | **エクスポート検証**。ONNXモデルの入出力が正しいか検証 |

#### to_onnx.py の処理フロー

```python
1. PPO.load() でモデル読み込み
2. PolicyWrapper でポリシーネットワークをラップ
3. torch.onnx.export() でONNX形式に変換
4. onnx.checker.check_model() で検証
5. 正規化統計をJSONで保存
```

---

### js/ （ブラウザ側）

#### js/ai/ （AI関連）

| ファイル | 説明 |
|---------|------|
| `RLAgent.js` | **RLエージェント**。ONNXモデルを使用してパドル操作を決定。観測ベクトル構築、フレームスキップ、自動発射を実装 |
| `ONNXInference.js` | **ONNX推論**。ONNX Runtime Webを使用したブラウザ上でのニューラルネットワーク推論 |
| `AIController.js` | **AI制御**。RLエージェントの有効/無効切り替え、UI連携 |

#### RLAgent.js の主要メソッド

```javascript
class RLAgent {
    async initialize()  // ONNXモデル読み込み
    async update(dt)    // 毎フレーム呼び出し、行動決定
    buildObservation()  // 216次元ベクトル構築
    applyAction(action) // パドル移動適用（0=左, 1=静止, 2=右）
}
```

#### ONNXInference.js の主要メソッド

```javascript
class ONNXInference {
    async load()           // モデル読み込み
    async predict(obs)     // 行動確率を返す
    async getAction(obs)   // 最良行動を返す（argmax）
    async sampleAction(obs) // 確率的にサンプリング
}
```

---

#### js/entities/ （ゲームエンティティ）

| ファイル | 説明 |
|---------|------|
| `Ball.js` | ボールクラス。位置、速度、反射、貫通モード |
| `Paddle.js` | パドルクラス。左右移動、ボール発射 |
| `Block.js` | ブロック基底クラス |
| `NormalBlock.js` | 通常ブロック（1ヒットで破壊） |
| `DurableBlock.js` | 耐久ブロック（複数ヒット必要） |
| `PowerUp.js` | パワーアップアイテム（落下、収集） |

---

#### js/managers/ （管理クラス）

| ファイル | 説明 |
|---------|------|
| `InputManager.js` | 入力管理。キーボード/マウス/タッチ入力を統一 |
| `SoundManager.js` | サウンド管理。BGM、効果音の再生 |
| `StageManager.js` | ステージ管理。ステージ読み込み、進行 |
| `StorageManager.js` | データ保存。ハイスコア、設定のlocalStorage保存 |

---

#### js/systems/ （ゲームシステム）

| ファイル | 説明 |
|---------|------|
| `CollisionSystem.js` | 衝突判定。ボール-ブロック、ボール-パドル、ボール-壁 |
| `PowerUpSystem.js` | パワーアップ処理。生成、落下、収集、効果適用 |
| `ScoreSystem.js` | スコア管理。コンボ計算、スコア加算 |
| `ParticleSystem.js` | パーティクル。ブロック破壊時のエフェクト |

---

#### js/ui/ （ユーザーインターフェース）

| ファイル | 説明 |
|---------|------|
| `HUD.js` | ヘッドアップディスプレイ。スコア、ライフ、時間表示 |
| `MenuScreen.js` | メニュー画面。ゲーム開始、設定 |
| `PauseScreen.js` | ポーズ画面 |
| `RankingScreen.js` | ランキング画面。ハイスコア表示 |

---

#### js/game/ （ゲームコア）

| ファイル | 説明 |
|---------|------|
| `Game.js` | **メインゲームクラス**。ゲームループ、状態管理、各システム統合 |
| `GameState.js` | ゲーム状態。MENU, PLAYING, PAUSED, GAME_OVER等の状態遷移 |

---

#### js/stages/ & js/config/

| ファイル | 説明 |
|---------|------|
| `StageData.js` | ステージデータ。全10ステージのブロック配置 |
| `Constants.js` | ゲーム定数。キャンバスサイズ、速度、色など |

---

### models/ （モデル保存先）

| ディレクトリ | 説明 |
|-------------|------|
| `checkpoints/` | 学習中のチェックポイント（10万ステップごと） |
| `best_model/` | 評価で最高性能だったモデル |
| `final_model/` | 学習終了時のモデル |
| `onnx/` | ブラウザ用ONNXモデル |

---

### logs/ （ログ）

TensorBoard形式のログ。以下のコマンドで可視化:

```bash
tensorboard --logdir logs
```

## 参考リンク

- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- [Gymnasium](https://gymnasium.farama.org/)
- [PyTorch](https://pytorch.org/)
