export type Locale = "zh-CN" | "en";
export type Localized = { "zh-CN": string; en: string };
export type Evidence = "PAPER" | "OFFICIAL CODE" | "REAL SAMPLE" | "CHECKPOINT" | "TEACHING";

export type Chapter = {
  act: number;
  stage: string;
  title: Localized;
  question: Localized;
  answer: Localized;
  layers: [Localized, Localized, Localized];
  formula: string;
  cue: Localized;
  evidence: Evidence;
  source: string;
};

export type Act = {
  id: number;
  label: Localized;
  short: Localized;
  range: [number, number];
};

const z = (zh: string, en: string): Localized => ({ "zh-CN": zh, en });

export const ACTS: Act[] = [
  { id: 0, label: z("第一幕 · 先看全局", "Act I · Orient"), short: z("任务", "Mission"), range: [0, 1] },
  { id: 1, label: z("第二幕 · 准备输入", "Act II · Prepare"), short: z("输入", "Inputs"), range: [2, 4] },
  { id: 2, label: z("第三幕 · Lift", "Act III · Lift"), short: z("图像", "Image"), range: [5, 7] },
  { id: 3, label: z("第四幕 · 几何", "Act IV · Geometry"), short: z("坐标", "Geometry"), range: [8, 9] },
  { id: 4, label: z("第五幕 · Splat", "Act V · Splat"), short: z("BEV", "BEV"), range: [10, 11] },
  { id: 5, label: z("第六幕 · 学习与推理", "Act VI · Learn & infer"), short: z("训练", "Learn"), range: [12, 13] },
  { id: 6, label: z("第七幕 · 证据与决策", "Act VII · Evidence & action"), short: z("验证", "Proof"), range: [14, 16] },
];

export const CHAPTERS: Chapter[] = [
  {
    act: 0,
    stage: "MISSION",
    title: z("LSS 到底要做什么？", "What is LSS trying to do?"),
    question: z("六张互不相同的透视图，怎样变成规划可以直接使用的统一空间？", "How can six incompatible perspective views become one space a planner can use?"),
    answer: z("LSS 接收任意数量的相机图像及其标定，把图像语义直接编码到自车坐标系中的鸟瞰网格。", "LSS receives any number of calibrated camera images and directly encodes their semantics into an ego-centric bird’s-eye-view grid."),
    layers: [
      z("输入不是一张拼接图，而是 N 张图像，以及每张图像自己的内参 K、相机到自车的 R 与 t。", "The input is not a stitched panorama: it is N images plus each camera’s K, camera-to-ego R and t."),
      z("本文公开 checkpoint 的具体任务是车辆 BEV 语义分割：每个 0.5 m 网格输出一个 vehicle logit。", "The public checkpoint performs vehicle BEV semantic segmentation: one vehicle logit per 0.5 m cell."),
      z("LiDAR 不进入网络。本站稍后显示 LiDAR，只用于核验相机推理是否落在正确空间。", "LiDAR never enters the network. It appears later only to verify where camera inference lands in space."),
    ],
    formula: "{Iₙ, Kₙ, Rₙ, tₙ}ₙ₌₁ᴺ  →  ŷBEV ∈ ℝᴮˣ¹ˣ²⁰⁰ˣ²⁰⁰",
    cue: z("先点选六张输入，再与右侧真实 checkpoint 输出对照。", "Inspect the six inputs, then compare them with the real checkpoint output."),
    evidence: "PAPER",
    source: "Paper Fig. 4 · official forward()",
  },
  {
    act: 0,
    stage: "FORWARD MAP",
    title: z("先走完一次，再拆开", "Walk the whole path once"),
    question: z("一帧数据在 LSS 中究竟依次经过哪些模块？", "What exactly happens to one sample, in order?"),
    answer: z("先看完整链路：图像预处理 → CamEncode → Lift → 几何反投影 → Splat → BevEncode → logits；之后再逐段放大。", "First see the complete chain: preprocessing → CamEncode → Lift → geometric unprojection → Splat → BevEncode → logits. We will zoom into each segment next."),
    layers: [
      z("可学习部分有两个主干：逐相机图像编码器与聚合后的 BEV 编码器。", "There are two learned backbones: a per-camera image encoder and a post-aggregation BEV encoder."),
      z("两者之间是带标定的 Lift–Splat 层：内容由网络产生，位置由几何决定。", "Between them sits the calibrated Lift–Splat layer: the network produces content; geometry determines location."),
      z("训练时最终 BEV loss 沿整条链路反传，因此 latent depth 也会被任务监督塑造。", "During training, the final BEV loss backpropagates through the whole path, shaping latent depth as well."),
    ],
    formula: "images → CamEncode → Lift → get_geometry → Splat → BevEncode → logits",
    cue: z("使用流水线顶部的阶段按钮跳转；当前只记住输入、几何桥、输出。", "Use the stage rail to jump; for now remember input, geometric bridge and output."),
    evidence: "OFFICIAL CODE",
    source: "models.py · get_voxels() · forward()",
  },
  {
    act: 1,
    stage: "DATALOADER",
    title: z("一个训练样本包含什么？", "What is inside one training sample?"),
    question: z("nuScenes 的图像、标定和监督标签分别从哪里来？", "Where do nuScenes images, calibration and supervision targets come from?"),
    answer: z("数据加载器读取相机 sample_data 与 calibrated_sensor；GT 由 3D vehicle boxes 变换到 LiDAR 时刻的 ego frame 后栅格化。", "The loader reads camera sample_data and calibrated_sensor records; GT is made by transforming 3D vehicle boxes into the LiDAR-time ego frame and rasterizing them."),
    layers: [
      z("SegmentationData 返回 imgs、rots、trans、intrins、post_rots、post_trans 与 binimg；不返回 LiDAR。", "SegmentationData returns imgs, rots, trans, intrins, post_rots, post_trans and binimg; it does not return LiDAR."),
      z("官方训练默认每个样本随机选 5/6 个相机，相当于在训练期做 camera dropout；验证时使用全部相机。", "Official training randomly chooses 5 of 6 cameras per sample, acting as camera dropout; validation uses every camera."),
      z("GT mask 是 200×200 的二值栅格。3D 框的底面角点经 ego pose 逆变换后填充到对应网格。", "The GT mask is a 200×200 binary raster. Bottom corners of 3D boxes are inverse ego-pose transformed and filled into grid cells."),
    ],
    formula: "sample → (images, calibration, post-transform, BEV target)",
    cue: z("切换相机，查看同一 sample 的真实图像与对应 K、R、t。", "Switch cameras to inspect the real image and its matching K, R and t."),
    evidence: "OFFICIAL CODE",
    source: "data.py · get_image_data() · get_binimg()",
  },
  {
    act: 1,
    stage: "CALIBRATION",
    title: z("矩阵不是魔法数字", "Matrices are not magic numbers"),
    question: z("K、R、t 和 ego pose 如何产生，又分别改变什么？", "Where do K, R, t and ego pose come from, and what does each change?"),
    answer: z("K 描述针孔相机，calibrated_sensor 的四元数与平移描述 camera→ego；ego_pose 描述某个传感器时刻的 ego→global。", "K describes the pinhole camera; calibrated_sensor quaternion and translation define camera→ego; ego_pose defines ego→global at a sensor timestamp."),
    layers: [
      z("四元数先归一化再转成正交旋转矩阵 R，平移 t 放入齐次矩阵最后一列。", "The quaternion is normalized into an orthogonal rotation R; translation t occupies the last homogeneous column."),
      z("列向量约定下，A→B→C 的组合是 Tᴮ→ᶜTᴬ→ᴮ；最先执行的矩阵写在最右。", "With column vectors, A→B→C composes as Tᴮ→ᶜTᴬ→ᴮ; the first operation sits on the right."),
      z("不同传感器时刻的 LiDAR→camera 还需经过两个 ego pose；不能用静态 inv(cam2ego)·lidar2ego 冒充。", "Cross-time LiDAR→camera also traverses two ego poses; static inv(cam2ego)·lidar2ego is not equivalent."),
    ],
    formula: "pᶜ = Tᴮ→ᶜ Tᴬ→ᴮ pᴬ · T⁻¹=[Rᵀ,−Rᵀt]",
    cue: z("点击坐标链上的边，查看来源、矩阵、逆矩阵与数值残差。", "Click any frame-graph edge to inspect provenance, matrix, inverse and residual."),
    evidence: "REAL SAMPLE",
    source: "nuScenes calibrated_sensor · ego_pose",
  },
  {
    act: 1,
    stage: "PREPROCESS",
    title: z("改图像，也要改坐标", "Transform the image and its coordinates"),
    question: z("resize、crop、flip、rotate 为什么必须留下几何记录？", "Why must resize, crop, flip and rotation leave a geometric record?"),
    answer: z("模型看的是增强后的 128×352 图像，但 K 定义在原图上；post_rot/post_trans 保存原图像素到网络像素的仿射变换。", "The model sees an augmented 128×352 image, while K belongs to the raw image; post_rot/post_trans record the raw-to-network affine transform."),
    layers: [
      z("固定验证帧由 1600×900 resize 到 352×198，再裁出 352×128；训练时还会随机缩放、裁剪、翻转和旋转。", "The pinned validation frame resizes 1600×900 to 352×198, then crops 352×128; training also randomizes scale, crop, flip and rotation."),
      z("RGB 随后转 tensor 并使用 ImageNet mean/std 归一化；这一步改变数值，不改变像素几何。", "RGB is then tensorized and ImageNet-normalized; this changes values, not pixel geometry."),
      z("进入 get_geometry 时必须先做 A⁻¹(u′−a)，把网络像素恢复到 K 所在的原图坐标。", "get_geometry must first apply A⁻¹(u′−a), returning network pixels to the raw-image frame where K is defined."),
    ],
    formula: "u′ = A u + a  ⇒  u = A⁻¹(u′−a)",
    cue: z("拖动前后对照，观察同一像素怎样随裁剪移动。", "Compare before and after to see how one pixel moves with the crop."),
    evidence: "REAL SAMPLE",
    source: "tools.py · img_transform() · normalize_img",
  },
  {
    act: 2,
    stage: "CAMENCODE",
    title: z("每个相机先独立理解图像", "Each camera is encoded independently"),
    question: z("128×352 的 RGB 如何变成可以 Lift 的 8×22 特征？", "How does 128×352 RGB become an 8×22 feature map ready to lift?"),
    answer: z("所有相机共享同一个 EfficientNet-B0。B 与 N 先合并，逐图提取多尺度特征，再恢复相机维度。", "All cameras share one EfficientNet-B0. B and N are merged for per-image multiscale encoding, then the camera dimension is restored."),
    layers: [
      z("EfficientNet reduction_5 上采样并与 reduction_4 拼接，经两层卷积形成 512 通道、8×22 的特征。", "EfficientNet reduction_5 is upsampled and concatenated with reduction_4, then two convolutions produce a 512-channel 8×22 map."),
      z("一个 1×1 depthnet 输出 D+C=41+64=105 个通道：前 41 是 depth logits，后 64 是 context。", "A 1×1 depthnet emits D+C=41+64=105 channels: 41 depth logits followed by 64 context channels."),
      z("相机之间此刻没有互相通信；共享权重让同一种视觉模式在任意相机上使用同一种编码规则。", "Cameras do not communicate yet; shared weights apply the same visual encoding rule to every camera."),
    ],
    formula: "[B,N,3,128,352] → [B·N,105,8,22]",
    cue: z("选择 context 通道，查看真实 checkpoint 在 8×22 网格上的响应。", "Select a context channel to inspect its real checkpoint response over the 8×22 grid."),
    evidence: "CHECKPOINT",
    source: "models.py · CamEncode · get_eff_depth()",
  },
  {
    act: 2,
    stage: "RAY",
    title: z("一个像素不是一个三维点", "A pixel is not a 3D point"),
    question: z("为什么 K⁻¹ 只能得到射线，不能直接得到位置？", "Why does K⁻¹ recover a ray but not a location?"),
    answer: z("单目像素只确定从光心出发的方向；沿这条射线的任意深度都会投影回同一个像素。", "A monocular pixel determines only a direction from the optical center; every depth along that ray reprojects to the same pixel."),
    layers: [
      z("相机坐标约定为 +x 向右、+y 向下、+z 沿光轴向前；LSS 的 depth d 就是 zcam。", "Camera coordinates use +x right, +y down and +z optical-forward; LSS depth d is zcam."),
      z("官方 frustum 在每个 8×22 anchor 上枚举 4…44 m，共 41 个深度假设。", "The official frustum enumerates 41 depth hypotheses from 4 to 44 m at every 8×22 anchor."),
      z("8×22 anchor 由 linspace 覆盖 0…351 与 0…127，不是简单的 16 像素方块中心。", "The 8×22 anchors are linspace samples over 0…351 and 0…127, not naïve 16-pixel block centers."),
    ],
    formula: "p_cam(d)=K⁻¹[d·u,d·v,d]ᵀ",
    cue: z("点击网络图像；站点会吸附到真实 feature anchor，并画出 41 个候选位置。", "Click the network image; the lab snaps to the real feature anchor and draws all 41 candidates."),
    evidence: "OFFICIAL CODE",
    source: "models.py · create_frustum()",
  },
  {
    act: 2,
    stage: "LIFT",
    title: z("用概率分配语义，而不是猜一个深度", "Distribute semantics instead of guessing one depth"),
    question: z("LSS 如何在不知道真实深度时构造三维特征？", "How does LSS construct 3D features without knowing true depth?"),
    answer: z("41 个 depth logits 经 softmax 得到 α(d)，再与同一位置的 64D context 做外积，把语义按概率铺到整条视锥。", "The 41 depth logits become α(d) through softmax, then outer-product with the 64D context to spread semantics along the frustum."),
    layers: [
      z("每个深度点携带 f[d,c]=α[d]·c[c]；不同深度共享 context 内容，只改变权重。", "Every depth point carries f[d,c]=α[d]·c[c]; depths share context content and differ only in weight."),
      z("softmax 让 41 个权重和为 1，但它不保证单峰，也不等价于物理深度概率。", "Softmax makes the 41 weights sum to one, but does not require a single mode or equal physical depth probability."),
      z("原始 LSS 没有深度真值 loss。α 是为最终 BEV 任务服务的 latent allocation，由 BEV 监督间接学习。", "Original LSS has no depth ground-truth loss. α is a task-oriented latent allocation learned indirectly from BEV supervision."),
    ],
    formula: "f(d,c) = softmax(ℓdepth)[d] · context[c]",
    cue: z("比较 checkpoint、one-hot、uniform 和多峰分布，观察同一语义怎样沿射线展开。", "Compare checkpoint, one-hot, uniform and multimodal depth to see how the same semantics spreads along the ray."),
    evidence: "CHECKPOINT",
    source: "models.py · get_depth_feat()",
  },
  {
    act: 3,
    stage: "GET_GEOMETRY",
    title: z("逐步执行 get_geometry", "Execute get_geometry step by step"),
    question: z("一个 [u′,v′,d] anchor 最终怎样落到 ego 米制坐标？", "How does one [u′,v′,d] anchor land in metric ego coordinates?"),
    answer: z("顺序固定：撤销图像增强 → 乘深度 → K⁻¹ → camera→ego 旋转 → 加 camera→ego 平移。", "The order is fixed: undo image augmentation → multiply by depth → K⁻¹ → camera→ego rotation → camera→ego translation."),
    layers: [
      z("第一步把网络坐标恢复到原图；第二步构造 [du,dv,d]，这才满足针孔投影的齐次比例。", "First return network coordinates to raw-image coordinates; then form [du,dv,d] to satisfy pinhole homogeneous scale."),
      z("K⁻¹ 得到相机坐标 [xcam,ycam,zcam]；R 改变坐标轴方向，t 把光心移动到车体中的真实位置。", "K⁻¹ yields [xcam,ycam,zcam]; R changes basis orientation and t moves the optical center to its real body location."),
      z("本站把同一个 anchor 的符号、真实数字、合并矩阵和官方代码绑定在一起，并用 18 个官方黄金点核验。", "The lab binds symbolic math, real numbers, composed matrix and official code to one anchor, checked against 18 official golden points."),
    ],
    formula: "pₑ = Rcam→ego K⁻¹[d·A⁻¹(u′−a), d]ᵀ + tcam→ego",
    cue: z("逐个点击五个运算节点；不要跳过中间坐标系。", "Advance through the five operations one at a time; do not skip frames."),
    evidence: "REAL SAMPLE",
    source: "models.py · get_geometry()",
  },
  {
    act: 3,
    stage: "MULTI-CAMERA",
    title: z("六个视锥进入同一个 ego", "Six frustums enter one ego frame"),
    question: z("来自不同相机的特征为什么能够落到同一个物体上？", "Why can features from different cameras land on the same object?"),
    answer: z("每台相机独立使用自己的 K、R、t，把候选点变换到同一个 ego frame；相机顺序不再重要，空间位置才重要。", "Each camera independently uses its own K, R and t to transform candidates into the shared ego frame; camera order stops mattering, spatial location does."),
    layers: [
      z("相机光心来自 cam2ego 的平移列，光轴来自旋转矩阵第三列；六个视锥都必须从车体向外。", "Camera centers come from the cam2ego translation column and optical axes from rotation column three; every frustum must point outward."),
      z("交换图像与对应标定的排列不会改变 sum pooling 结果，这就是相机排列不变性。", "Permuting images together with their calibration leaves sum pooling unchanged: camera-permutation invariance."),
      z("若刚体移动整个 ego 系统，输出应做同样移动；几何显式提供这种 ego 等距等变结构。", "Rigidly moving the whole ego system should move the output the same way; explicit geometry provides ego-isometry equivariance."),
    ],
    formula: "Gₙ = Tcamₙ→ego · Kₙ⁻¹ · frustumₙ",
    cue: z("依次选择六台相机，检查光心与 +z 光轴；它们应全部朝车外。", "Select all six cameras and inspect center and +z optical axis; each must face outward."),
    evidence: "REAL SAMPLE",
    source: "Paper Sec. 3 · real calibrated rig",
  },
  {
    act: 4,
    stage: "SPLAT",
    title: z("把稀疏候选点泼到规则网格", "Splat candidates into a regular grid"),
    question: z("成千上万的视锥特征怎样变成 CNN 可以处理的 BEV tensor？", "How do tens of thousands of frustum features become a BEV tensor a CNN can process?"),
    answer: z("Splat 将 ego 米制坐标量化为 voxel index，过滤越界点，把同一 pillar 的特征精确求和。", "Splat quantizes metric ego coordinates into voxel indices, filters out-of-range points and exactly sums features within each pillar."),
    layers: [
      z("六相机共有 6·41·8·22=43,296 个候选点；网格使用半开区间 [−50,50)，分辨率 0.5 m。", "Six cameras produce 6·41·8·22=43,296 candidates; the grid is half-open [−50,50) at 0.5 m resolution."),
      z("[ix,iy,iz,b] 编码成 rank 并排序，让同一 voxel 的特征在内存中相邻。", "[ix,iy,iz,b] is encoded into a rank and sorted so features in one voxel become adjacent in memory."),
      z("QuickCumsum 用前缀和、组末保留和相邻差分得到 exact sum；反向时同组点收到同一聚合梯度。", "QuickCumsum uses prefix sums, group-end retention and adjacent differences for exact sums; backward sends the aggregate gradient to every point in the group."),
    ],
    formula: "index = floor((pₑ − (b−Δ/2)) / Δ)  ·  BEVcell = Σ features",
    cue: z("点击 BEV cell，反查六台相机各有多少 frustum samples 落入其中。", "Click a BEV cell to trace how many frustum samples each camera contributed."),
    evidence: "OFFICIAL CODE",
    source: "models.py · voxel_pooling() · tools.py · QuickCumsum",
  },
  {
    act: 4,
    stage: "BEVENCODE",
    title: z("几何结束，二维推理开始", "Geometry ends; 2D reasoning begins"),
    question: z("Splat 之后为什么还需要一个 BEV CNN？", "Why is a BEV CNN still needed after splatting?"),
    answer: z("Splat 只负责把证据放到正确位置；BevEncode 在规则网格上补充上下文、融合邻域并输出任务 logits。", "Splat only puts evidence in the right place; BevEncode adds context, fuses neighborhoods and emits task logits on the regular grid."),
    layers: [
      z("聚合先写入 [B,C,Z,X,Y]；原配置只有一个高 20 m 的 z voxel，因此 collapse 后是 [B,64,200,200]。", "Pooling first writes [B,C,Z,X,Y]; the original config has one 20 m-tall z voxel, so collapse yields [B,64,200,200]."),
      z("ResNet-18 风格 trunk 下采样到多尺度，再把 layer3 ×4 上采样并与 layer1 拼接。", "A ResNet-18-style trunk builds multiscale context, then upsamples layer3 ×4 and concatenates layer1."),
      z("最后再 ×2 上采样，以 1×1 convolution 输出 [B,1,200,200] raw logits；此时尚未 sigmoid。", "A final ×2 upsample and 1×1 convolution produce [B,1,200,200] raw logits; sigmoid has not yet been applied."),
    ],
    formula: "[B,64,200,200] → ResNet-18/FPN-like decoder → [B,1,200,200]",
    cue: z("在 logits、概率与 GT 之间切换；注意几何 tensor 与语义输出不是同一个东西。", "Switch between logits, probability and GT; the geometric tensor and semantic output are not the same object."),
    evidence: "OFFICIAL CODE",
    source: "models.py · BevEncode",
  },
  {
    act: 5,
    stage: "SUPERVISION",
    title: z("最终标签如何教会 latent depth？", "How does the final label teach latent depth?"),
    question: z("没有深度真值，41 个 depth weights 为什么仍会学出有用结构？", "Without depth labels, why do the 41 depth weights learn useful structure?"),
    answer: z("GT 只在 BEV 输出端计算 BCEWithLogits；梯度穿过 BevEncode、sum pooling 和外积，一路回到 depth logits 与图像 backbone。", "GT is used only for BCEWithLogits at the BEV output; gradients pass through BevEncode, sum pooling and the outer product back to depth logits and the image backbone."),
    layers: [
      z("一个错误 BEV cell 会惩罚所有向它贡献特征的 frustum points；softmax 和 context 因此共同调整。", "An incorrect BEV cell penalizes every frustum point that contributed to it, so softmax depth and context adjust together."),
      z("官方代码默认 Adam lr=1e−3、weight decay=1e−7、gradient clip=5，并随机使用 5 台相机。", "Official code defaults to Adam lr=1e−3, weight decay=1e−7, gradient clip=5 and randomly uses 5 cameras."),
      z("论文写 object segmentation positive weight=1.0；公开 train.py 默认 pos_weight=2.13。站点明确保留这一论文/代码差异。", "The paper states object positive weight=1.0; public train.py defaults to pos_weight=2.13. The lab preserves this paper/code discrepancy."),
    ],
    formula: "L = BCEWithLogits(ŷBEV,yBEV)  → ∂L/∂α(d), ∂L/∂context",
    cue: z("沿反向箭头逐层查看：一个 BEV 错误如何回传到选中像素的 depth distribution。", "Follow the backward arrows to see how one BEV error reaches the selected pixel’s depth distribution."),
    evidence: "OFFICIAL CODE",
    source: "train.py · SimpleLoss · Paper Sec. 5.1",
  },
  {
    act: 5,
    stage: "INFERENCE",
    title: z("推理与后处理到底有哪些？", "What are inference and post-processing?"),
    question: z("从六张新图像到最终可显示结果，网络之后还需要做什么？", "After the network sees six new images, what remains before a displayable result?"),
    answer: z("验证预处理是确定性的；forward 输出 logits，sigmoid 转为概率，阈值产生二值 vehicle mask。原始 LSS 分割任务到此结束。", "Validation preprocessing is deterministic; forward returns logits, sigmoid makes probabilities and a threshold creates a binary vehicle mask. That ends the original LSS segmentation task."),
    layers: [
      z("官方 IoU 代码直接使用 logits>0，这与 sigmoid(logit)>0.5 完全等价。", "Official IoU code uses logits>0, exactly equivalent to sigmoid(logit)>0.5."),
      z("原模型不输出 3D 检测框，因此不存在 box decode、NMS、tracking 或速度估计；这些属于后继系统。", "The original model does not output 3D boxes, so there is no box decode, NMS, tracking or velocity estimation; those belong to later systems."),
      z("显示阶段再把 tensor [x,y] 映射到屏幕：+x 向上、+y 向左。这个显示变换不能回写到模型数据。", "Display then maps tensor [x,y] to screen with +x up and +y left. This display transform must never mutate model data."),
    ],
    formula: "prob = σ(logit) · mask = prob ≥ τ · official τ=0.5",
    cue: z("拖动阈值，观察 TP/FP/FN 与单帧 IoU 如何变化；这不是论文验证集指标。", "Move the threshold and observe TP/FP/FN and single-frame IoU; this is not the paper’s validation metric."),
    evidence: "CHECKPOINT",
    source: "tools.py · get_batch_iou()",
  },
  {
    act: 6,
    stage: "TRUTH LAB",
    title: z("怎样判断 BEV 没有画反？", "How do we know the BEV is not mirrored?"),
    question: z("相机图看起来不会像 BEV 热图，那应该用什么建立对应？", "A camera image will never resemble a BEV heatmap, so what should establish correspondence?"),
    answer: z("用对象级和点级 linked evidence：同一个 LiDAR 点、GT 车辆框或 BEV cell 在相机、3D 与 BEV 中共享 ID 和坐标链。", "Use object-level and point-level linked evidence: the same LiDAR point, GT box or BEV cell shares an ID and coordinate chain across camera, 3D and BEV."),
    layers: [
      z("透视像素先沿深度分配，再由六相机聚合，二维轮廓本来就不会被保持。", "Perspective pixels are distributed over depth and aggregated across cameras, so their 2D silhouette is not preserved."),
      z("本站固定显示 34,688 个真实 LiDAR 点；它们从不参与 checkpoint forward，只是独立空间尺。", "The lab displays 34,688 real LiDAR points; they never participate in checkpoint forward and serve only as an independent spatial ruler."),
      z("模型概率、GT、LiDAR occupancy、相机覆盖与屏幕坐标都使用同一 ego→BEV→screen 函数。", "Model probability, GT, LiDAR occupancy, camera coverage and screen coordinates all use the same ego→BEV→screen functions."),
    ],
    formula: "LiDAR id ↔ camera uv ↔ ego xyz ↔ BEV [ix,iy]",
    cue: z("点击相机中的 LiDAR 点，再核对 3D 高亮、BEV index 与完整投影数字。", "Click a LiDAR point in the camera and verify the 3D highlight, BEV index and full projection numbers."),
    evidence: "REAL SAMPLE",
    source: "Pinned nuScenes sample · OpenMMLab transforms",
  },
  {
    act: 6,
    stage: "RESULTS",
    title: z("论文证据说明了什么？", "What does the paper evidence show?"),
    question: z("单帧演示之外，怎样评价 LSS 是否真的有效和鲁棒？", "Beyond one demo frame, how do we judge whether LSS is effective and robust?"),
    answer: z("论文在 nuScenes/Lyft 的车辆与地图分割上比较 CNN、Frozen Encoder、OFT，并系统测试 camera dropout、标定噪声和新相机阵列。", "The paper compares CNN, Frozen Encoder and OFT on nuScenes/Lyft object and map segmentation, then tests camera dropout, calibration noise and unseen camera rigs."),
    layers: [
      z("nuScenes 车辆 IoU：CNN 24.25、Frozen 26.83、OFT 30.05、Lift-Splat 32.07。", "nuScenes vehicle IoU: CNN 24.25, Frozen 26.83, OFT 30.05 and Lift-Splat 32.07."),
      z("随机丢相机和加入外参噪声进行训练能提高对应故障下的鲁棒性；不是推理时自动免疫。", "Training with camera dropout and extrinsic noise improves robustness to those failures; inference is not automatically immune."),
      z("oracle-depth PointPillars 仍更强；相机模型在夜间与远距离下降明显，论文认为多时间步是重要方向。", "Oracle-depth PointPillars remains stronger; camera performance drops at night and range, and the paper identifies multi-timestep input as important future work."),
    ],
    formula: "dataset evidence ≠ one-frame visualization",
    cue: z("关闭任一相机或加入 FRONT yaw 扰动，对比缓存的真实 checkpoint 输出。", "Drop one camera or perturb FRONT yaw and compare cached real checkpoint outputs."),
    evidence: "PAPER",
    source: "Paper Tables 1–5 · Figs. 6–10",
  },
  {
    act: 6,
    stage: "SHOOT",
    title: z("从 BEV 表征到动作", "From BEV representation to action"),
    question: z("“Shoot” 是普通后处理，还是另一个被监督的任务？", "Is “Shoot” ordinary post-processing, or another supervised task?"),
    answer: z("Shoot 是论文中的规划头：BEV 网络输出 cost map，1K 条轨迹模板在其上累积代价，再形成 Boltzmann 分布。", "Shoot is the paper’s planning head: the BEV network predicts a cost map, 1K trajectory templates accumulate its costs, then form a Boltzmann distribution."),
    layers: [
      z("1K 模板由训练集 expert trajectories 做 K-Means 得到，每条轨迹 5 秒、采样间隔 0.25 秒。", "The 1K templates come from K-Means over training expert trajectories; each spans 5 seconds at 0.25-second intervals."),
      z("GT 轨迹先匹配最近模板，再用 cross entropy 训练；测试时选择概率最高的模板。", "The GT trajectory is matched to its nearest template and trained with cross entropy; inference selects the highest-probability template."),
      z("官方仓库没有发布规划 checkpoint。本站只重建方程与代表性模板，绝不把教学 cost map 标成真实模型结果。", "The official repository released no planning checkpoint. This lab reconstructs only the equation and representative templates, never labeling its teaching cost map as model output."),
    ],
    formula: "p(τᵢ|o)=exp(−Σ₍ₓ,ᵧ₎∈τᵢ cₒ(x,y)) / Σⱼ exp(−cost(τⱼ))",
    cue: z("调整温度并悬停轨迹；最后沿顶部流水线反向复述完整 LSS。", "Adjust temperature and hover trajectories, then recite the full LSS path backward along the rail."),
    evidence: "TEACHING",
    source: "Paper Eq. 2 · Sec. 3.3 and 5.6",
  },
];

export function tx(locale: Locale, value: Localized) {
  return value[locale];
}

export function actForChapter(index: number) {
  return ACTS.find((act) => index >= act.range[0] && index <= act.range[1]) ?? ACTS[0];
}
