
ô
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8£â£@£â£H£â£b<cond_1/then/_10/cond_1/Adam/Adam/update_56/ResourceApplyAdamhuZUÖB
˚
ï_ZN13cutlass_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi64ELi256ELi32EEENS4_51Conv2dWgradOutputGradientTileAccessIteratorAnalyticINS_11MatrixShapeILi64ELi32EEENS_6half_tENS_9transform29PitchLinearWarpRakedThreadMapINS_6layout16PitchLinearShapeILi64ELi32EEELi128ENSG_ILi8ELi4EEELi8EEEEENSD_11threadblock25RegularTileAccessIteratorISB_SC_NSF_40ColumnMajorTensorOpMultiplicandCongruousILi16ELi64EEELi1ESJ_Li16EEELNS_4arch14CacheOperation4KindE0ENS4_47Conv2dWgradActivationTileAccessIteratorAnalyticINSA_ILi32ELi256EEESC_NSE_INSG_ILi256ELi32EEELi128ESI_Li8EEEEENSM_ISU_SC_NSF_37RowMajorTensorOpMultiplicandCongruousILi16ELi64EEELi0ESW_Li16EEELSS_0ENS6_11threadblock9MmaPolicyINS6_4warp11MmaTensorOpINS7_ILi64ELi64ELi32EEESC_SO_SC_SZ_fNSF_8RowMajorENS13_17MmaTensorOpPolicyINSQ_3MmaINS7_ILi16ELi8ELi16EEELi32ESC_S16_SC_NSF_11ColumnMajorEfS16_NSQ_13OpMultiplyAddEEENSA_ILi1ELi1EEEEELi1ELb0EbEENSA_ILi0ELi0EEES1G_Li1EEELi4EbEENS_8epilogue11threadblock8EpilogueIS8_S1F_Li1ENS1K_22PredicatedTileIteratorINS1K_26OutputTileOptimalThreadMapINS1K_15OutputTileShapeILi256ELi8ELi1ELi1ELi1EEENS1O_ILi1ELi8ELi1ELi1ELi8EEELi128ELi4ELi32EEEfEENS1J_4warp24FragmentIteratorTensorOpIS15_S19_fNS_5ArrayIfLi4ELb1EEES16_EENS1T_20TileIteratorTensorOpIS15_S19_fS16_EENS1K_18SharedLoadIteratorINS1R_18CompactedThreadMapEfLi16EEENS1J_6thread17LinearCombinationIfLi4EffLNS_15FloatRoundStyleE2EEENSA_ILi0ELi8EEEEENS11_30GemmIdentityThreadblockSwizzleILi4EEELNS1_8OperatorE2ENS1_17Conv2dProblemSizeEEEEEvNT_6ParamsEˇ ÄÄ*Ä22)8‚òì@‚òìH‚òìXb<gradient_tape/model_5/conv2d_131/Conv2D/Conv2DBackpropFilterh
ú
∂void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride>(cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride::Params)ı Ä¿*Ä2Ä8‘ÿÿ@‘ÿÿH‘ÿÿXb;gradient_tape/model_5/conv2d_131/Conv2D/Conv2DBackpropInputh
¡
|sm80_xmma_fprop_implicit_gemm_f16f16_f16f32_f32_nhwckrsc_nhwc_tilesize256x64x32_stage3_warpsize4x1x1_g1_tensor16x8x16_kernel˙ Ä‡*Ä2Ä8”÷@”÷H”÷PXbmodel_5/conv2d_131/Conv2Dh
≠
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8¡‡ç@¡‡çH¡‡çbAdam/gradients/AddN_31huZUÖB
ú
∂void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride>(cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride::Params)ı Ä¿*Ä2Ä8¡†ç@¡†çH¡†çXb;gradient_tape/model_5/conv2d_137/Conv2D/Conv2DBackpropInputh
‡
úvoid cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3>(cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3::Params)˛ Ä¿*Ä2Ä8¡»ã@¡»ãH¡»ãXbmodel_5/conv2d_137/Conv2Dh
x
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ä¿8øÿÑ@øÿÑHøÿÑb/gradient_tape/dense_15/kernel/Regularizer/Mul_1huZUÖB
∆
ﬁvoid xmma_cudnn::gemm::kernel<xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, xmma_cudnn::Col, 256, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 3> >(xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, xmma_cudnn::Col, 256, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 3>::Params)Ó Ä¿*Ä2$	8øÿÇ@øÿÇHøÿÇPXb<gradient_tape/model_5/conv2d_130/Conv2D/Conv2DBackpropFilterh
˚
ï_ZN13cutlass_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi64ELi256ELi32EEENS4_51Conv2dWgradOutputGradientTileAccessIteratorAnalyticINS_11MatrixShapeILi64ELi32EEENS_6half_tENS_9transform29PitchLinearWarpRakedThreadMapINS_6layout16PitchLinearShapeILi64ELi32EEELi128ENSG_ILi8ELi4EEELi8EEEEENSD_11threadblock25RegularTileAccessIteratorISB_SC_NSF_40ColumnMajorTensorOpMultiplicandCongruousILi16ELi64EEELi1ESJ_Li16EEELNS_4arch14CacheOperation4KindE0ENS4_47Conv2dWgradActivationTileAccessIteratorAnalyticINSA_ILi32ELi256EEESC_NSE_INSG_ILi256ELi32EEELi128ESI_Li8EEEEENSM_ISU_SC_NSF_37RowMajorTensorOpMultiplicandCongruousILi16ELi64EEELi0ESW_Li16EEELSS_0ENS6_11threadblock9MmaPolicyINS6_4warp11MmaTensorOpINS7_ILi64ELi64ELi32EEESC_SO_SC_SZ_fNSF_8RowMajorENS13_17MmaTensorOpPolicyINSQ_3MmaINS7_ILi16ELi8ELi16EEELi32ESC_S16_SC_NSF_11ColumnMajorEfS16_NSQ_13OpMultiplyAddEEENSA_ILi1ELi1EEEEELi1ELb0EbEENSA_ILi0ELi0EEES1G_Li1EEELi4EbEENS_8epilogue11threadblock8EpilogueIS8_S1F_Li1ENS1K_22PredicatedTileIteratorINS1K_26OutputTileOptimalThreadMapINS1K_15OutputTileShapeILi256ELi8ELi1ELi1ELi1EEENS1O_ILi1ELi8ELi1ELi1ELi8EEELi128ELi4ELi32EEEfEENS1J_4warp24FragmentIteratorTensorOpIS15_S19_fNS_5ArrayIfLi4ELb1EEES16_EENS1T_20TileIteratorTensorOpIS15_S19_fS16_EENS1K_18SharedLoadIteratorINS1R_18CompactedThreadMapEfLi16EEENS1J_6thread17LinearCombinationIfLi4EffLNS_15FloatRoundStyleE2EEENSA_ILi0ELi8EEEEENS11_30GemmIdentityThreadblockSwizzleILi4EEELNS1_8OperatorE2ENS1_17Conv2dProblemSizeEEEEEvNT_6ParamsEˇ ÄÄ*Ä2
8Ω‡¸@Ω‡¸HΩ‡¸Xb<gradient_tape/model_5/conv2d_137/Conv2D/Conv2DBackpropFilterh
‡
úvoid cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3>(cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3::Params)˛ Ä¿*Ä2Ä8Ωò˚@Ωò˚HΩò˚Xbmodel_5/conv2d_130/Conv2Dh
ú
∂void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride>(cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride::Params)ı Ä¿*Ä2Ä8º–˘@º–˘Hº–˘Xb;gradient_tape/model_5/conv2d_130/Conv2D/Conv2DBackpropInputh
‡
úvoid cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3>(cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3::Params)˛ Ä¿*Ä2Ä8¥∞’@¥∞’H¥∞’Xbmodel_5/conv2d_134/Conv2Dh
˚
ï_ZN13cutlass_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi64ELi256ELi32EEENS4_51Conv2dWgradOutputGradientTileAccessIteratorAnalyticINS_11MatrixShapeILi64ELi32EEENS_6half_tENS_9transform29PitchLinearWarpRakedThreadMapINS_6layout16PitchLinearShapeILi64ELi32EEELi128ENSG_ILi8ELi4EEELi8EEEEENSD_11threadblock25RegularTileAccessIteratorISB_SC_NSF_40ColumnMajorTensorOpMultiplicandCongruousILi16ELi64EEELi1ESJ_Li16EEELNS_4arch14CacheOperation4KindE0ENS4_47Conv2dWgradActivationTileAccessIteratorAnalyticINSA_ILi32ELi256EEESC_NSE_INSG_ILi256ELi32EEELi128ESI_Li8EEEEENSM_ISU_SC_NSF_37RowMajorTensorOpMultiplicandCongruousILi16ELi64EEELi0ESW_Li16EEELSS_0ENS6_11threadblock9MmaPolicyINS6_4warp11MmaTensorOpINS7_ILi64ELi64ELi32EEESC_SO_SC_SZ_fNSF_8RowMajorENS13_17MmaTensorOpPolicyINSQ_3MmaINS7_ILi16ELi8ELi16EEELi32ESC_S16_SC_NSF_11ColumnMajorEfS16_NSQ_13OpMultiplyAddEEENSA_ILi1ELi1EEEEELi1ELb0EbEENSA_ILi0ELi0EEES1G_Li1EEELi4EbEENS_8epilogue11threadblock8EpilogueIS8_S1F_Li1ENS1K_22PredicatedTileIteratorINS1K_26OutputTileOptimalThreadMapINS1K_15OutputTileShapeILi256ELi8ELi1ELi1ELi1EEENS1O_ILi1ELi8ELi1ELi1ELi8EEELi128ELi4ELi32EEEfEENS1J_4warp24FragmentIteratorTensorOpIS15_S19_fNS_5ArrayIfLi4ELb1EEES16_EENS1T_20TileIteratorTensorOpIS15_S19_fS16_EENS1K_18SharedLoadIteratorINS1R_18CompactedThreadMapEfLi16EEENS1J_6thread17LinearCombinationIfLi4EffLNS_15FloatRoundStyleE2EEENSA_ILi0ELi8EEEEENS11_30GemmIdentityThreadblockSwizzleILi4EEELNS1_8OperatorE2ENS1_17Conv2dProblemSizeEEEEEvNT_6ParamsEˇ ÄÄ*Ä2
8±†À@±†ÀH±†ÀXb<gradient_tape/model_5/conv2d_134/Conv2D/Conv2DBackpropFilterh
‡
úvoid cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3>(cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3::Params)˛ Ä¿*Ä2Ä8Ø¿@Ø¿HØ¿Xbmodel_5/conv2d_142/Conv2Dh
ú
∂void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride>(cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride::Params)ı Ä¿*Ä2Ä8Æ∞º@Æ∞ºHÆ∞ºXb;gradient_tape/model_5/conv2d_134/Conv2D/Conv2DBackpropInputh
∆
ﬁvoid xmma_cudnn::gemm::kernel<xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, xmma_cudnn::Col, 256, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 3> >(xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, xmma_cudnn::Col, 256, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 3>::Params)Ó Ä¿*Ä2Z8´ÿ≥@´ÿ≥H´ÿ≥PXb<gradient_tape/model_5/conv2d_142/Conv2D/Conv2DBackpropFilterh
¥
Œvoid cudnn::ops::pooling_bw_kernel_max<__half, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, false>(cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor) Ä *Ä2Ä 8´∞≥@´∞≥H´∞≥b:gradient_tape/model_5/max_pooling2d_21/MaxPool/MaxPoolGradhu  »B
O
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ä¿8™Æ@™ÆH™Æbmul_58huZUÖB
n
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ä¿8™¿Æ@™¿ÆH™¿Æb"dense_15/kernel/Regularizer/SquarehuZUÖB
v
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ä¿8™†Æ@™†ÆH™†Æb-gradient_tape/dense_15/kernel/Regularizer/MulhuZUÖB
∫
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8©à≠@©à≠H©à≠b/gradient_tape/model_5/dense_15/MatMul/Cast/CasthuZUÖB
≥
Œvoid cudnn::ops::pooling_bw_kernel_max<__half, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, false>(cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor) Ä *Ä2   8©»©@©»©H©»©b:gradient_tape/model_5/max_pooling2d_20/MaxPool/MaxPoolGradhu  »B
ú
∂void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride>(cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride::Params)ı Ä¿*Ä2Ä8¶Ëö@¶ËöH¶ËöXb;gradient_tape/model_5/conv2d_142/Conv2D/Conv2DBackpropInputh
≥
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8¶Äö@¶ÄöH¶Äöbmodel_5/dense_15/MatMul/CasthuZUÖB
˚
ï_ZN13cutlass_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi64ELi256ELi32EEENS4_51Conv2dWgradOutputGradientTileAccessIteratorAnalyticINS_11MatrixShapeILi64ELi32EEENS_6half_tENS_9transform29PitchLinearWarpRakedThreadMapINS_6layout16PitchLinearShapeILi64ELi32EEELi128ENSG_ILi8ELi4EEELi8EEEEENSD_11threadblock25RegularTileAccessIteratorISB_SC_NSF_40ColumnMajorTensorOpMultiplicandCongruousILi16ELi64EEELi1ESJ_Li16EEELNS_4arch14CacheOperation4KindE0ENS4_47Conv2dWgradActivationTileAccessIteratorAnalyticINSA_ILi32ELi256EEESC_NSE_INSG_ILi256ELi32EEELi128ESI_Li8EEEEENSM_ISU_SC_NSF_37RowMajorTensorOpMultiplicandCongruousILi16ELi64EEELi0ESW_Li16EEELSS_0ENS6_11threadblock9MmaPolicyINS6_4warp11MmaTensorOpINS7_ILi64ELi64ELi32EEESC_SO_SC_SZ_fNSF_8RowMajorENS13_17MmaTensorOpPolicyINSQ_3MmaINS7_ILi16ELi8ELi16EEELi32ESC_S16_SC_NSF_11ColumnMajorEfS16_NSQ_13OpMultiplyAddEEENSA_ILi1ELi1EEEEELi1ELb0EbEENSA_ILi0ELi0EEES1G_Li1EEELi4EbEENS_8epilogue11threadblock8EpilogueIS8_S1F_Li1ENS1K_22PredicatedTileIteratorINS1K_26OutputTileOptimalThreadMapINS1K_15OutputTileShapeILi256ELi8ELi1ELi1ELi1EEENS1O_ILi1ELi8ELi1ELi1ELi8EEELi128ELi4ELi32EEEfEENS1J_4warp24FragmentIteratorTensorOpIS15_S19_fNS_5ArrayIfLi4ELb1EEES16_EENS1T_20TileIteratorTensorOpIS15_S19_fS16_EENS1K_18SharedLoadIteratorINS1R_18CompactedThreadMapEfLi16EEENS1J_6thread17LinearCombinationIfLi4EffLNS_15FloatRoundStyleE2EEENSA_ILi0ELi8EEEEENS11_30GemmIdentityThreadblockSwizzleILi4EEELNS1_8OperatorE2ENS1_17Conv2dProblemSizeEEEEEvNT_6ParamsEˇ ÄÄ*Ä2 8†–Ö@†–ÖH†–ÖXb<gradient_tape/model_5/conv2d_143/Conv2D/Conv2DBackpropFilterh
¿
|sm80_xmma_fprop_implicit_gemm_f16f16_f16f32_f32_nhwckrsc_nhwc_tilesize256x64x32_stage3_warpsize4x1x1_g1_tensor16x8x16_kernel˙ Ä‡*Ä2 8°¿Ö@°¿ÖH°¿ÖPXbmodel_5/conv2d_143/Conv2Dh
ô
∂void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride>(cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride::Params)ı Ä¿*Ä2Ä8ö¯k@ö¯kHö¯kXb;gradient_tape/model_5/conv2d_143/Conv2D/Conv2DBackpropInputh
÷
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8ôi@ôiHôib.gradient_tape/dense_15/kernel/Regularizer/TilehuZUÖB
U
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2ÄÄ8ö»i@ö»iHö»ibIsFinite_56hu  »B
√
ﬁvoid xmma_cudnn::gemm::kernel<xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 32, 2, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 32, 2, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 32, 2, 2, 1, 1>, 16, xmma_cudnn::Col, 128, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 32, 2, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 32, 2, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 4> >(xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 32, 2, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 32, 2, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 32, 2, 2, 1, 1>, 16, xmma_cudnn::Col, 128, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 32, 2, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 32, 2, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 4>::Params)Ù ÄÄ*Ä2	8ô∞g@ô∞gHô∞gPXb<gradient_tape/model_5/conv2d_128/Conv2D/Conv2DBackpropFilterh
ô
∂void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride>(cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride::Params)ı Ä¿*Ä2Ä8ò»c@ò»cHò»cXb;gradient_tape/model_5/conv2d_136/Conv2D/Conv2DBackpropInputh
¯
ï_ZN13cutlass_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi64ELi256ELi32EEENS4_51Conv2dWgradOutputGradientTileAccessIteratorAnalyticINS_11MatrixShapeILi64ELi32EEENS_6half_tENS_9transform29PitchLinearWarpRakedThreadMapINS_6layout16PitchLinearShapeILi64ELi32EEELi128ENSG_ILi8ELi4EEELi8EEEEENSD_11threadblock25RegularTileAccessIteratorISB_SC_NSF_40ColumnMajorTensorOpMultiplicandCongruousILi16ELi64EEELi1ESJ_Li16EEELNS_4arch14CacheOperation4KindE0ENS4_47Conv2dWgradActivationTileAccessIteratorAnalyticINSA_ILi32ELi256EEESC_NSE_INSG_ILi256ELi32EEELi128ESI_Li8EEEEENSM_ISU_SC_NSF_37RowMajorTensorOpMultiplicandCongruousILi16ELi64EEELi0ESW_Li16EEELSS_0ENS6_11threadblock9MmaPolicyINS6_4warp11MmaTensorOpINS7_ILi64ELi64ELi32EEESC_SO_SC_SZ_fNSF_8RowMajorENS13_17MmaTensorOpPolicyINSQ_3MmaINS7_ILi16ELi8ELi16EEELi32ESC_S16_SC_NSF_11ColumnMajorEfS16_NSQ_13OpMultiplyAddEEENSA_ILi1ELi1EEEEELi1ELb0EbEENSA_ILi0ELi0EEES1G_Li1EEELi4EbEENS_8epilogue11threadblock8EpilogueIS8_S1F_Li1ENS1K_22PredicatedTileIteratorINS1K_26OutputTileOptimalThreadMapINS1K_15OutputTileShapeILi256ELi8ELi1ELi1ELi1EEENS1O_ILi1ELi8ELi1ELi1ELi8EEELi128ELi4ELi32EEEfEENS1J_4warp24FragmentIteratorTensorOpIS15_S19_fNS_5ArrayIfLi4ELb1EEES16_EENS1T_20TileIteratorTensorOpIS15_S19_fS16_EENS1K_18SharedLoadIteratorINS1R_18CompactedThreadMapEfLi16EEENS1J_6thread17LinearCombinationIfLi4EffLNS_15FloatRoundStyleE2EEENSA_ILi0ELi8EEEEENS11_30GemmIdentityThreadblockSwizzleILi4EEELNS1_8OperatorE2ENS1_17Conv2dProblemSizeEEEEEvNT_6ParamsEˇ ÄÄ*Ä2
8ò¯a@ò¯aHò¯aXb<gradient_tape/model_5/conv2d_136/Conv2D/Conv2DBackpropFilterh
›
úvoid cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3>(cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3::Params)˛ Ä¿*Ä2Ä8óa@óaHóaXbmodel_5/conv2d_136/Conv2Dh
›
úvoid cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3>(cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3::Params)˛ Ä¿*Ä2Ä8ó∏_@ó∏_Hó∏_Xbmodel_5/conv2d_128/Conv2Dh
≥
Œvoid xmma_cudnn::gemm::kernel<xmma_cudnn::implicit_gemm::dgrad::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_a_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 16, false, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 16, xmma_cudnn::Row, 32, 256> >, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_c_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, 16, xmma_cudnn::Fragment_c<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, false>, false>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 3> >(xmma_cudnn::implicit_gemm::dgrad::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_a_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 16, false, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 16, xmma_cudnn::Row, 32, 256> >, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_c_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, 16, xmma_cudnn::Fragment_c<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, false>, false>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 3>::Params)ˇ Ä‡*Ä2Ä8ó‡]@ó‡]Hó‡]PXb;gradient_tape/model_5/conv2d_128/Conv2D/Conv2DBackpropInputh
¬
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2ú8î∞S@î∞SHî∞Sbdense_15/kernel/Regularizer/Sumhu  »B
ò
∂void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride>(cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride::Params)ı Ä¿*Ä2@8ìÄQ@ìÄQHìÄQXb;gradient_tape/model_5/conv2d_140/Conv2D/Conv2DBackpropInputh
›
úvoid cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3>(cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3::Params)˛ Ä¿*Ä2Ä8ì¿N@ì¿NHì¿NXbmodel_5/conv2d_133/Conv2Dh
¯
ï_ZN13cutlass_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi64ELi256ELi32EEENS4_51Conv2dWgradOutputGradientTileAccessIteratorAnalyticINS_11MatrixShapeILi64ELi32EEENS_6half_tENS_9transform29PitchLinearWarpRakedThreadMapINS_6layout16PitchLinearShapeILi64ELi32EEELi128ENSG_ILi8ELi4EEELi8EEEEENSD_11threadblock25RegularTileAccessIteratorISB_SC_NSF_40ColumnMajorTensorOpMultiplicandCongruousILi16ELi64EEELi1ESJ_Li16EEELNS_4arch14CacheOperation4KindE0ENS4_47Conv2dWgradActivationTileAccessIteratorAnalyticINSA_ILi32ELi256EEESC_NSE_INSG_ILi256ELi32EEELi128ESI_Li8EEEEENSM_ISU_SC_NSF_37RowMajorTensorOpMultiplicandCongruousILi16ELi64EEELi0ESW_Li16EEELSS_0ENS6_11threadblock9MmaPolicyINS6_4warp11MmaTensorOpINS7_ILi64ELi64ELi32EEESC_SO_SC_SZ_fNSF_8RowMajorENS13_17MmaTensorOpPolicyINSQ_3MmaINS7_ILi16ELi8ELi16EEELi32ESC_S16_SC_NSF_11ColumnMajorEfS16_NSQ_13OpMultiplyAddEEENSA_ILi1ELi1EEEEELi1ELb0EbEENSA_ILi0ELi0EEES1G_Li1EEELi4EbEENS_8epilogue11threadblock8EpilogueIS8_S1F_Li1ENS1K_22PredicatedTileIteratorINS1K_26OutputTileOptimalThreadMapINS1K_15OutputTileShapeILi256ELi8ELi1ELi1ELi1EEENS1O_ILi1ELi8ELi1ELi1ELi8EEELi128ELi4ELi32EEEfEENS1J_4warp24FragmentIteratorTensorOpIS15_S19_fNS_5ArrayIfLi4ELb1EEES16_EENS1T_20TileIteratorTensorOpIS15_S19_fS16_EENS1K_18SharedLoadIteratorINS1R_18CompactedThreadMapEfLi16EEENS1J_6thread17LinearCombinationIfLi4EffLNS_15FloatRoundStyleE2EEENSA_ILi0ELi8EEEEENS11_30GemmIdentityThreadblockSwizzleILi4EEELNS1_8OperatorE2ENS1_17Conv2dProblemSizeEEEEEvNT_6ParamsEˇ ÄÄ*Ä2
8íÿI@íÿIHíÿIXb<gradient_tape/model_5/conv2d_133/Conv2D/Conv2DBackpropFilterh
¯
ï_ZN13cutlass_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi64ELi256ELi32EEENS4_51Conv2dWgradOutputGradientTileAccessIteratorAnalyticINS_11MatrixShapeILi64ELi32EEENS_6half_tENS_9transform29PitchLinearWarpRakedThreadMapINS_6layout16PitchLinearShapeILi64ELi32EEELi128ENSG_ILi8ELi4EEELi8EEEEENSD_11threadblock25RegularTileAccessIteratorISB_SC_NSF_40ColumnMajorTensorOpMultiplicandCongruousILi16ELi64EEELi1ESJ_Li16EEELNS_4arch14CacheOperation4KindE0ENS4_47Conv2dWgradActivationTileAccessIteratorAnalyticINSA_ILi32ELi256EEESC_NSE_INSG_ILi256ELi32EEELi128ESI_Li8EEEEENSM_ISU_SC_NSF_37RowMajorTensorOpMultiplicandCongruousILi16ELi64EEELi0ESW_Li16EEELSS_0ENS6_11threadblock9MmaPolicyINS6_4warp11MmaTensorOpINS7_ILi64ELi64ELi32EEESC_SO_SC_SZ_fNSF_8RowMajorENS13_17MmaTensorOpPolicyINSQ_3MmaINS7_ILi16ELi8ELi16EEELi32ESC_S16_SC_NSF_11ColumnMajorEfS16_NSQ_13OpMultiplyAddEEENSA_ILi1ELi1EEEEELi1ELb0EbEENSA_ILi0ELi0EEES1G_Li1EEELi4EbEENS_8epilogue11threadblock8EpilogueIS8_S1F_Li1ENS1K_22PredicatedTileIteratorINS1K_26OutputTileOptimalThreadMapINS1K_15OutputTileShapeILi256ELi8ELi1ELi1ELi1EEENS1O_ILi1ELi8ELi1ELi1ELi8EEELi128ELi4ELi32EEEfEENS1J_4warp24FragmentIteratorTensorOpIS15_S19_fNS_5ArrayIfLi4ELb1EEES16_EENS1T_20TileIteratorTensorOpIS15_S19_fS16_EENS1K_18SharedLoadIteratorINS1R_18CompactedThreadMapEfLi16EEENS1J_6thread17LinearCombinationIfLi4EffLNS_15FloatRoundStyleE2EEENSA_ILi0ELi8EEEEENS11_30GemmIdentityThreadblockSwizzleILi4EEELNS1_8OperatorE2ENS1_17Conv2dProblemSizeEEEEEvNT_6ParamsEˇ ÄÄ*Ä2	8ë®I@ë®IHë®IXb<gradient_tape/model_5/conv2d_140/Conv2D/Conv2DBackpropFilterh
√
ﬁvoid xmma_cudnn::gemm::kernel<xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, xmma_cudnn::Col, 256, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 3> >(xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, xmma_cudnn::Col, 256, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 3>::Params)Ó Ä¿*Ä2-8ë¯G@ë¯GHë¯GPXb<gradient_tape/model_5/conv2d_139/Conv2D/Conv2DBackpropFilterh
ô
∂void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride>(cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride::Params)ı Ä¿*Ä2Ä8ëÿE@ëÿEHëÿEXb;gradient_tape/model_5/conv2d_133/Conv2D/Conv2DBackpropInputh
›
úvoid cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3>(cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3::Params)˛ Ä¿*Ä2Ä8ë–D@ë–DHë–DXbmodel_5/conv2d_127/Conv2Dh
≥
Œvoid xmma_cudnn::gemm::kernel<xmma_cudnn::implicit_gemm::dgrad::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_a_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 16, false, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 16, xmma_cudnn::Row, 32, 256> >, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_c_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, 16, xmma_cudnn::Fragment_c<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, false>, false>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 3> >(xmma_cudnn::implicit_gemm::dgrad::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_a_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 16, false, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 16, xmma_cudnn::Row, 32, 256> >, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_c_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, 16, xmma_cudnn::Fragment_c<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, false>, false>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 3>::Params)ˇ Ä‡*Ä2Ä8ë®D@ë®DHë®DPXb;gradient_tape/model_5/conv2d_127/Conv2D/Conv2DBackpropInputh
Ω
|sm80_xmma_fprop_implicit_gemm_f16f16_f16f32_f32_nhwckrsc_nhwc_tilesize256x64x32_stage3_warpsize4x1x1_g1_tensor16x8x16_kernel˙ Ä‡*Ä2 8êòC@êòCHêòCPXbmodel_5/conv2d_140/Conv2Dh
√
ﬁvoid xmma_cudnn::gemm::kernel<xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, xmma_cudnn::Col, 256, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 3> >(xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, xmma_cudnn::Col, 256, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 3>::Params)Ó Ä¿*Ä2		8ê‡A@ê‡AHê‡APXb<gradient_tape/model_5/conv2d_127/Conv2D/Conv2DBackpropFilterh
ê
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2ÄÄ8é‡<@Ñ∏HÖ‡bAgradient_tape/model_5/batch_normalization_52/FusedBatchNormGradV3hu  »B
ò
∂void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride>(cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride::Params)ı Ä¿*Ä2@8èÿ:@èÿ:Hèÿ:Xb;gradient_tape/model_5/conv2d_139/Conv2D/Conv2DBackpropInputh
ê
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2ÄÄ8é»:@Ñ∏HÖ»bAgradient_tape/model_5/batch_normalization_53/FusedBatchNormGradV3hu  »B
ê
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2ÄÄ8é†:@Ö∞HÖ∏bAgradient_tape/model_5/batch_normalization_51/FusedBatchNormGradV3hu  »B
Ì
Évoid cudnn::bn_bw_1C11_kernel_new<__half, float, float2, 512, true, 1>(float, float, float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, float const*, float*, float*, float const*, float const*, float)(ê*Ä2Ä8éË8@éË8HéË8bAgradient_tape/model_5/batch_normalization_51/FusedBatchNormGradV3hu  »B
±
Œvoid cudnn::ops::pooling_bw_kernel_max<__half, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, false>(cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor) Ä *Ä2† 8éò7@éò7Héò7b:gradient_tape/model_5/max_pooling2d_22/MaxPool/MaxPoolGradhu  »B
Ì
Évoid cudnn::bn_bw_1C11_kernel_new<__half, float, float2, 512, true, 1>(float, float, float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, float const*, float*, float*, float const*, float const*, float)(ê*Ä2Ä8ç®6@ç®6Hç®6bAgradient_tape/model_5/batch_normalization_53/FusedBatchNormGradV3hu  »B
Ì
Évoid cudnn::bn_bw_1C11_kernel_new<__half, float, float2, 512, true, 1>(float, float, float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, float const*, float*, float*, float const*, float const*, float)(ê*Ä2Ä8é†6@é†6Hé†6bAgradient_tape/model_5/batch_normalization_52/FusedBatchNormGradV3hu  »B
›
úvoid cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3>(cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3::Params)˛ Ä¿*Ä2Ä8å‡0@å‡0Hå‡0Xbmodel_5/conv2d_139/Conv2Dh
~
-ampere_fp16_s1688gemm_fp16_256x64_ldg8_f2f_ntÍÄ¿*Ä2Ä8ãË-@ãË-HãË-b'gradient_tape/model_5/dense_15/MatMul_1hugUÖA
{
:ampere_fp16_s16816gemm_fp16_128x64_ldg8_f2f_stages_64x4_nnñÄÄ ÄÄ*Ä28ã∏-@ã∏-Hã∏-Xbmodel_5/dense_15/MatMulh
ä
:ampere_fp16_s16816gemm_fp16_128x64_ldg8_f2f_stages_64x4_tnûÄÄ ÄÄ*Ä2Ä	8ã,@ã,Hã,Xb%gradient_tape/model_5/dense_15/MatMulh
±
Œvoid cudnn::ops::pooling_bw_kernel_max<__half, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, false>(cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor) Ä *Ä2† 8ä)@ä)Hä)b:gradient_tape/model_5/max_pooling2d_23/MaxPool/MaxPoolGradhu  »B
˛
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2ÄÄ8âà'@Ñ¿HÖ»b/model_5/batch_normalization_51/FusedBatchNormV3hu  »B
˛
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2ÄÄ8âà'@Ö∏HÑ–b/model_5/batch_normalization_53/FusedBatchNormV3hu  »B
˛
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2ÄÄ8âÄ'@Ö∏HÑ»b/model_5/batch_normalization_52/FusedBatchNormV3hu  »B
Ÿ

ü
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8âÄ&@âÄ&HâÄ&bAdam/gradients/AddN_8huZUÖB
·
ùvoid convolve_common_engine_float_NHWC<__half, __half, 128, 5, 5, 3, 3, 3, true, false, false, false, false>(int, int, int, __half const*, __half const*, int, __half*, conv_kernel_common_params, unsigned long long, unsigned long, float, float, bool, __half const*, __half const*, bool)PÄ*2ÄÄ8à– @à– Hà– Xbmodel_5/conv2d_125/Conv2Dhu  HB
∞
ÿvoid cudnn::bn_fw_tr_1C11_kernel_NCHW<__half, float, 512, true, 1>(cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(ê*Ä2Ä8á¿ @á¿ Há¿ b/model_5/batch_normalization_53/FusedBatchNormV3hu  »B
∞
ÿvoid cudnn::bn_fw_tr_1C11_kernel_NCHW<__half, float, 512, true, 1>(cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(ê*Ä2Ä8á® @á® Há® b/model_5/batch_normalization_52/FusedBatchNormV3hu  »B
¯
ï_ZN13cutlass_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi64ELi256ELi32EEENS4_51Conv2dWgradOutputGradientTileAccessIteratorAnalyticINS_11MatrixShapeILi64ELi32EEENS_6half_tENS_9transform29PitchLinearWarpRakedThreadMapINS_6layout16PitchLinearShapeILi64ELi32EEELi128ENSG_ILi8ELi4EEELi8EEEEENSD_11threadblock25RegularTileAccessIteratorISB_SC_NSF_40ColumnMajorTensorOpMultiplicandCongruousILi16ELi64EEELi1ESJ_Li16EEELNS_4arch14CacheOperation4KindE0ENS4_47Conv2dWgradActivationTileAccessIteratorAnalyticINSA_ILi32ELi256EEESC_NSE_INSG_ILi256ELi32EEELi128ESI_Li8EEEEENSM_ISU_SC_NSF_37RowMajorTensorOpMultiplicandCongruousILi16ELi64EEELi0ESW_Li16EEELSS_0ENS6_11threadblock9MmaPolicyINS6_4warp11MmaTensorOpINS7_ILi64ELi64ELi32EEESC_SO_SC_SZ_fNSF_8RowMajorENS13_17MmaTensorOpPolicyINSQ_3MmaINS7_ILi16ELi8ELi16EEELi32ESC_S16_SC_NSF_11ColumnMajorEfS16_NSQ_13OpMultiplyAddEEENSA_ILi1ELi1EEEEELi1ELb0EbEENSA_ILi0ELi0EEES1G_Li1EEELi4EbEENS_8epilogue11threadblock8EpilogueIS8_S1F_Li1ENS1K_22PredicatedTileIteratorINS1K_26OutputTileOptimalThreadMapINS1K_15OutputTileShapeILi256ELi8ELi1ELi1ELi1EEENS1O_ILi1ELi8ELi1ELi1ELi8EEELi128ELi4ELi32EEEfEENS1J_4warp24FragmentIteratorTensorOpIS15_S19_fNS_5ArrayIfLi4ELb1EEES16_EENS1T_20TileIteratorTensorOpIS15_S19_fS16_EENS1K_18SharedLoadIteratorINS1R_18CompactedThreadMapEfLi16EEENS1J_6thread17LinearCombinationIfLi4EffLNS_15FloatRoundStyleE2EEENSA_ILi0ELi8EEEEENS11_30GemmIdentityThreadblockSwizzleILi4EEELNS1_8OperatorE2ENS1_17Conv2dProblemSizeEEEEEvNT_6ParamsEˇ ÄÄ*Ä2)8à‡@à‡Hà‡Xb<gradient_tape/model_5/conv2d_125/Conv2D/Conv2DBackpropFilterh
Ö
√void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*Ä2{8á–@Ç†HÉêbmodel_5/concatenate_41/concathuZUÖB
∞
ÿvoid cudnn::bn_fw_tr_1C11_kernel_NCHW<__half, float, 512, true, 1>(cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(ê*Ä2Ä8à∞@à∞Hà∞b/model_5/batch_normalization_51/FusedBatchNormV3hu  »B
∫
htensorflow::functor::ReluGradHalfKernelVector(Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2ÄÄ8áê@áêHáêb,gradient_tape/model_5/activation_57/ReluGradhu  »B
∫
htensorflow::functor::ReluGradHalfKernelVector(Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2ÄÄ8áà@áàHáàb,gradient_tape/model_5/activation_56/ReluGradhu  »B
∫
htensorflow::functor::ReluGradHalfKernelVector(Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2ÄÄ8á¯@á¯Há¯b,gradient_tape/model_5/activation_58/ReluGradhu  »B
Ö
√void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*Ä2{8á‡@Ç†HÉêbmodel_5/concatenate_42/concathuZUÖB
Ö
√void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*Ä2{8áÿ@Ç†HÑêbmodel_5/concatenate_43/concathuZUÖB
ñ
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8á¯@á¯Há¯b<cond_1/then/_10/cond_1/Adam/Adam/update_50/ResourceApplyAdamhuZUÖB
°
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_tn‚ÄÄ ÄÄ*Ä2Ä8Üò@ÜòHÜòXb;gradient_tape/model_5/conv2d_129/Conv2D/Conv2DBackpropInputh
å
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2ú8Öê@ÖêHÖêbAll_56hu  »B
¶	
Ávoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ö®@Ö®HÖ®bmodel_5/activation_56/ReluhuZUÖB
~
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_nnﬁÄÄ ÄÄ*Ä2@8Ö†@Ö†HÖ†Xbmodel_5/conv2d_141/Conv2Dh
¶	
Ávoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Öê@ÖêHÖêbmodel_5/activation_57/ReluhuZUÖB
¶	
Ávoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*Ä2{8ÖÄ@ÖÄHÖÄbmodel_5/activation_58/ReluhuZUÖB
≠
Àvoid cutlass_cudnn::Kernel<cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4> >(cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4>::Params)6* 2 28Ö®@Ö®HÖ®Xb<gradient_tape/model_5/conv2d_131/Conv2D/Conv2DBackpropFilterhu  ñB
†
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_tn‚ÄÄ ÄÄ*Ä2
@8ÑÄ@ÑÄHÑÄXb;gradient_tape/model_5/conv2d_141/Conv2D/Conv2DBackpropInputh
ê
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2Ä†8Öÿ@ÇòHÇ†bAgradient_tape/model_5/batch_normalization_54/FusedBatchNormGradV3hu  »B
ê
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2Ä†8Ñ»@ÅòHÇòbAgradient_tape/model_5/batch_normalization_55/FusedBatchNormGradV3hu  »B
ñ
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8Ñê@ÑêHÑêb<cond_1/then/_10/cond_1/Adam/Adam/update_52/ResourceApplyAdamhuZUÖB
°
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_64x3_ntÊÄÄ ÄÄ*Ä2
8Ö»@Ö»HÖ»Xb<gradient_tape/model_5/conv2d_141/Conv2D/Conv2DBackpropFilterh
ê
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2Äê8ÑË@Å»HÇ–bAgradient_tape/model_5/batch_normalization_57/FusedBatchNormGradV3hu  »B

;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_nnﬁÄÄ ÄÄ*Ä2Ä8Ñ–@Ñ–HÑ–Xbmodel_5/conv2d_129/Conv2Dh
ö
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Ñ†@Ñ†HÑ†b,gradient_tape/model_5/concatenate_41/Slice_2huZUÖB
°
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_64x3_ntÊÄÄ ÄÄ*Ä28Ñ¯@Ñ¯HÑ¯Xb<gradient_tape/model_5/conv2d_129/Conv2D/Conv2DBackpropFilterh
ö
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Ñ¯@Ñ¯HÑ¯b,gradient_tape/model_5/concatenate_42/Slice_1huZUÖB
ö
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Ñ@ÑHÑb,gradient_tape/model_5/concatenate_43/Slice_1huZUÖB
Ì
Évoid cudnn::bn_bw_1C11_kernel_new<__half, float, float2, 128, true, 1>(float, float, float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, float const*, float*, float*, float const*, float const*, float)(ê*Ä2Ä8ÑË@ÑËHÑËbAgradient_tape/model_5/batch_normalization_54/FusedBatchNormGradV3hu  »B
Ì
Évoid cudnn::bn_bw_1C11_kernel_new<__half, float, float2, 128, true, 1>(float, float, float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, float const*, float*, float*, float const*, float const*, float)(ê*Ä2Ä8ÑË@ÑËHÑËbAgradient_tape/model_5/batch_normalization_55/FusedBatchNormGradV3hu  »B
™
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8É®@É®HÉ®bmodel_5/conv2d_125/BiasAddhuZUÖB
™
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Ñò@ÑòHÑòbmodel_5/conv2d_127/BiasAddhuZUÖB
™
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Éò@ÉòHÉòbmodel_5/conv2d_130/BiasAddhuZUÖB
”
åvoid pooling_fw_kernel_max_nhwc<__half, float, 0, (cudnnNanPropagation_t)0, false>(PoolingParams, __half*, __half const*, float, float, int)*Ä2ÄÄ8É¿@É¿HÉ¿b model_5/max_pooling2d_21/MaxPoolhu  »B
”
åvoid pooling_fw_kernel_max_nhwc<__half, float, 0, (cudnnNanPropagation_t)0, false>(PoolingParams, __half*, __half const*, float, float, int)*Ä2ÄÄ8Éê@ÉêHÉêb model_5/max_pooling2d_20/MaxPoolhu  »B
≠
Àvoid cutlass_cudnn::Kernel<cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4> >(cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4>::Params)6* 2@}8É¯@É¯HÉ¯Xb<gradient_tape/model_5/conv2d_143/Conv2D/Conv2DBackpropFilterhu  ñB
—
Ô_ZN13cutlass_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi64ELi64ELi32EEENS4_51Conv2dWgradOutputGradientTileAccessIteratorAnalyticINS_11MatrixShapeILi64ELi32EEENS_6half_tENS_9transform29PitchLinearWarpRakedThreadMapINS_6layout16PitchLinearShapeILi64ELi32EEELi128ENSG_ILi8ELi4EEELi8EEEEENSD_11threadblock25RegularTileAccessIteratorISB_SC_NSF_40ColumnMajorTensorOpMultiplicandCongruousILi16ELi64EEELi1ESJ_Li16EEELNS_4arch14CacheOperation4KindE0ENS4_47Conv2dWgradActivationTileAccessIteratorAnalyticINSA_ILi32ELi64EEESC_SJ_EENSM_ISU_SC_NSF_37RowMajorTensorOpMultiplicandCongruousILi16ELi64EEELi0ESJ_Li16EEELSS_0ENS6_11threadblock9MmaPolicyINS6_4warp11MmaTensorOpINS7_ILi32ELi32ELi32EEESC_SO_SC_SX_fNSF_8RowMajorENS11_17MmaTensorOpPolicyINSQ_3MmaINS7_ILi16ELi8ELi16EEELi32ESC_S14_SC_NSF_11ColumnMajorEfS14_NSQ_13OpMultiplyAddEEENSA_ILi1ELi1EEEEELi1ELb0EbEENSA_ILi0ELi0EEES1E_Li1EEELi10EbEENS_8epilogue11threadblock8EpilogueIS8_S1D_Li1ENS1I_22PredicatedTileIteratorINS1I_26OutputTileOptimalThreadMapINS1I_15OutputTileShapeILi64ELi8ELi2ELi1ELi1EEENS1M_ILi1ELi4ELi1ELi1ELi4EEELi128ELi4ELi32EEEfEENS1H_4warp24FragmentIteratorTensorOpIS13_S17_fNS_5ArrayIfLi4ELb1EEES14_EENS1R_20TileIteratorTensorOpIS13_S17_fS14_EENS1I_18SharedLoadIteratorINS1P_18CompactedThreadMapEfLi16EEENS1H_6thread17LinearCombinationIfLi4EffLNS_15FloatRoundStyleE2EEENSA_ILi0ELi8EEEEENSZ_30GemmIdentityThreadblockSwizzleILi4EEELNS1_8OperatorE2ENS1_17Conv2dProblemSizeEEEEEvNT_6ParamsEh ÄÄ*Ä2)8ÉË@ÉËHÉËXb<gradient_tape/model_5/conv2d_124/Conv2D/Conv2DBackpropFilterh
˛
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2Ä†8Ñ¿@Ç†HÇ†b/model_5/batch_normalization_55/FusedBatchNormV3hu  »B
˛
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2Ä†8Ç¿@ÅòHÅ®b/model_5/batch_normalization_54/FusedBatchNormV3hu  »B
≠
Àvoid cutlass_cudnn::Kernel<cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4> >(cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4>::Params)6* 2@?8É∏@É∏HÉ∏Xb<gradient_tape/model_5/conv2d_137/Conv2D/Conv2DBackpropFilterhu  ñB
Ÿ

ü
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8É¯@É¯HÉ¯bAdam/gradients/AddN_6huZUÖB
≠
Àvoid cutlass_cudnn::Kernel<cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4> >(cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4>::Params)6* 2@?8Ç®@Ç®HÇ®Xb<gradient_tape/model_5/conv2d_140/Conv2D/Conv2DBackpropFilterhu  ñB
˛
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2Äê8Ç†@Å–HÅ–b/model_5/batch_normalization_57/FusedBatchNormV3hu  »B
™
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ç¿
@Ç¿
HÇ¿
bAdam/gradients/AddN_29huZUÖB
v
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2¿8É∞
@É∞
HÉ∞
b1gradient_tape/conv2d_142/kernel/Regularizer/Mul_1huZUÖB
ü
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8É∞
@É∞
HÉ∞
b4gradient_tape/model_5/conv2d_125/BiasAdd/BiasAddGradhuZUÖB
˙
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8É®
@É®
HÉ®
Xb<gradient_tape/model_5/conv2d_142/Conv2D/Conv2DBackpropFilterhuZUÖB
°
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_64x3_ntÊÄÄ ÄÄ*Ä28Ç†
@Ç†
HÇ†
Xb<gradient_tape/model_5/conv2d_135/Conv2D/Conv2DBackpropFilterh
ü
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Éò
@Éò
HÉò
b4gradient_tape/model_5/conv2d_127/BiasAdd/BiasAddGradhuZUÖB
°
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_64x3_ntÊÄÄ ÄÄ*Ä28Çò
@Çò
HÇò
Xb<gradient_tape/model_5/conv2d_132/Conv2D/Conv2DBackpropFilterh
ü
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Éà
@Éà
HÉà
b4gradient_tape/model_5/conv2d_130/BiasAdd/BiasAddGradhuZUÖB
≠
Àvoid cutlass_cudnn::Kernel<cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4> >(cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4>::Params)6* 2@28ÉÄ
@ÉÄ
HÉÄ
Xb<gradient_tape/model_5/conv2d_134/Conv2D/Conv2DBackpropFilterhu  ñB
ü
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8ÉÄ
@ÉÄ
HÉÄ
b4gradient_tape/model_5/conv2d_124/BiasAdd/BiasAddGradhuZUÖB
ü
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Ç¯	@Ç¯	HÇ¯	b4gradient_tape/model_5/conv2d_123/BiasAdd/BiasAddGradhuZUÖB
è
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2ÄP8É‡	@Å†HÅ†bAgradient_tape/model_5/batch_normalization_56/FusedBatchNormGradV3hu  »B
Ÿ

ü
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ç–	@Ç–	HÇ–	bAdam/gradients/AddN_7huZUÖB
Ÿ

ü
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ç¿	@Ç¿	HÇ¿	bAdam/gradients/AddN_9huZUÖB
Ö
√void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*Ä2{8É∞	@Å¯HÅ‡bmodel_5/concatenate_44/concathuZUÖB
Æ
Nvoid cudnn::ops::scalePackedTensor_kernel<__half, float>(long, __half*, float)*Ä2ˇˇ8É®	@É®	HÉ®	b:gradient_tape/model_5/max_pooling2d_21/MaxPool/MaxPoolGradhu  »B
Æ
Nvoid cudnn::ops::scalePackedTensor_kernel<__half, float>(long, __half*, float)*Ä2ˇˇ8Ç®	@Ç®	HÇ®	b:gradient_tape/model_5/max_pooling2d_20/MaxPool/MaxPoolGradhu  »B
Ö
√void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*Ä2{8Ç†	@Ä¯HÅÿbmodel_5/concatenate_45/concathuZUÖB
π
htensorflow::functor::ReluGradHalfKernelVector(Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2Ä(8Çò	@Çò	HÇò	b,gradient_tape/model_5/activation_59/ReluGradhu  »B
◊
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Çò	@Çò	HÇò	Xbmodel_5/conv2d_142/Conv2DhuZUÖB
˘
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Çê	@Çê	HÇê	Xb;gradient_tape/model_5/conv2d_142/Conv2D/Conv2DBackpropInputhuZUÖB
π
htensorflow::functor::ReluGradHalfKernelVector(Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2Ä(8Éà	@Éà	HÉà	b,gradient_tape/model_5/activation_60/ReluGradhu  »B
ñ
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8Çà	@Çà	HÇà	b<cond_1/then/_10/cond_1/Adam/Adam/update_60/ResourceApplyAdamhuZUÖB
·
ùvoid convolve_common_engine_float_NHWC<__half, __half, 128, 5, 5, 3, 3, 3, true, false, false, false, false>(int, int, int, __half const*, __half const*, int, __half*, conv_kernel_common_params, unsigned long long, unsigned long, float, float, bool, __half const*, __half const*, bool)PÄ*2ÄÄ8ÇÄ	@ÇÄ	HÇÄ	Xbmodel_5/conv2d_124/Conv2Dhu  HB
ñ
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8Ç‡@Ç‡HÇ‡b<cond_1/then/_10/cond_1/Adam/Adam/update_44/ResourceApplyAdamhuZUÖB
ˇ
ívoid cudnn::bn_bw_1C11_singleread_specialized<__half2, 512, 1, 2, 7>(float, float, float, float, cudnnTensorStruct, __half2 const*, cudnnTensorStruct, __half2 const*, cudnnTensorStruct, __half2*, float const*, float*, float*, float const*, float const*, float, cudnn::reduced_divisor, int, cudnn::reduced_divisor, cudnn::bnBwPersistentState*, int, float, float, float, int, float, cudnnStatus_t*, bool)8ê Ä *Ä2Ä8Ç–@Ç–HÇ–bAgradient_tape/model_5/batch_normalization_57/FusedBatchNormGradV3huZUÖB
ñ
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8Ç»@Ç»HÇ»b<cond_1/then/_10/cond_1/Adam/Adam/update_36/ResourceApplyAdamhuZUÖB
ÿ
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2{8Ç¿@Ç¿HÇ¿b0gradient_tape/conv2d_142/kernel/Regularizer/TilehuZUÖB
Ö
√void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*Ä2{8Å†@ÄÄHÅ–bmodel_5/concatenate_47/concathuZUÖB
π
htensorflow::functor::ReluGradHalfKernelVector(Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2Ä$8Çò@ÇòHÇòb,gradient_tape/model_5/activation_62/ReluGradhu  »B
ò
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Çò@ÇòHÇòb*gradient_tape/model_5/concatenate_41/SlicehuZUÖB
ö
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Çò@ÇòHÇòb,gradient_tape/model_5/concatenate_41/Slice_1huZUÖB
ò
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Çà@ÇàHÇàb*gradient_tape/model_5/concatenate_42/SlicehuZUÖB
ö
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Çà@ÇàHÇàb,gradient_tape/model_5/concatenate_42/Slice_2huZUÖB
ò
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Çà@ÇàHÇàb*gradient_tape/model_5/concatenate_43/SlicehuZUÖB
ö
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8ÇÄ@ÇÄHÇÄb,gradient_tape/model_5/concatenate_43/Slice_2huZUÖB
°
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_tn‚ÄÄ ÄÄ*Ä2Ä8Ç¯@Ç¯HÇ¯Xb;gradient_tape/model_5/conv2d_135/Conv2D/Conv2DBackpropInputh
q
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä 8Å¿@Å¿HÅ¿b.gradient_tape/model_5/dropout_31/dropout/Mul_1huZUÖB
c
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä 8Ç∏@Ç∏HÇ∏b model_5/dropout_31/dropout/Mul_1huZUÖB
c
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä 8Å∏@Å∏HÅ∏b model_5/dropout_30/dropout/Mul_1huZUÖB
q
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä 8Å∞@Å∞HÅ∞b.gradient_tape/model_5/dropout_30/dropout/Mul_1huZUÖB
™
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Ç®@Ç®HÇ®bmodel_5/conv2d_124/BiasAddhuZUÖB
™
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å†@Å†HÅ†bAdam/gradients/AddN_30huZUÖB
™
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Å†@Å†HÅ†bmodel_5/conv2d_126/BiasAddhuZUÖB
v
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2–8Çò@ÇòHÇòb1gradient_tape/conv2d_143/kernel/Regularizer/Mul_1huZUÖB
™
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Çò@ÇòHÇòbmodel_5/conv2d_123/BiasAddhuZUÖB
™
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Çò@ÇòHÇòbmodel_5/conv2d_129/BiasAddhuZUÖB
ñ
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8Çò@ÇòHÇòb<cond_1/then/_10/cond_1/Adam/Adam/update_28/ResourceApplyAdamhuZUÖB
™
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Åò@ÅòHÅòbmodel_5/conv2d_128/BiasAddhuZUÖB
™
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Çê@ÇêHÇêbmodel_5/conv2d_131/BiasAddhuZUÖB
t
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2¿8ÇÄ@ÇÄHÇÄb/gradient_tape/conv2d_142/kernel/Regularizer/MulhuZUÖB
l
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2¿8ÇÄ@ÇÄHÇÄb$conv2d_142/kernel/Regularizer/SquarehuZUÖB
K
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2¿8ÅÄ@ÅÄHÅÄbmul_52huZUÖB
◊
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8ÅË@ÅËHÅËXbmodel_5/conv2d_143/Conv2DhuZUÖB
—
ıvoid cudnn::bn_fw_tr_1C11_singleread_specialized<__half2, 512, 1, 2, 20>(cudnnTensorStruct, __half2 const*, cudnnTensorStruct, __half2*, float const*, float const*, float, float, float*, float*, float*, float*, float, float, cudnn::reduced_divisor, int, cudnn::reduced_divisor, cudnn::bnFwPersistentState*, int, float, float, float, int, float, float, cudnnStatus_t*, bool)@ê Ä¿*Ä2Ä8Ç–@Ç–HÇ–b/model_5/batch_normalization_54/FusedBatchNormV3huZUÖB
˘
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ç–@Ç–HÇ–Xb;gradient_tape/model_5/conv2d_143/Conv2D/Conv2DBackpropInputhuZUÖB
—
ıvoid cudnn::bn_fw_tr_1C11_singleread_specialized<__half2, 512, 1, 2, 20>(cudnnTensorStruct, __half2 const*, cudnnTensorStruct, __half2*, float const*, float const*, float, float, float*, float*, float*, float*, float, float, cudnn::reduced_divisor, int, cudnn::reduced_divisor, cudnn::bnFwPersistentState*, int, float, float, float, int, float, float, cudnnStatus_t*, bool)@ê Ä¿*Ä2Ä8Å»@Å»HÅ»b/model_5/batch_normalization_55/FusedBatchNormV3huZUÖB

;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_nnﬁÄÄ ÄÄ*Ä2Ä8Å¿@Å¿HÅ¿Xbmodel_5/conv2d_135/Conv2Dh
±
Ìvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ç∏@Ç∏HÇ∏bmodel_5/dropout_31/dropout/CasthuZUÖB
°
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_tn‚ÄÄ ÄÄ*Ä2Ä8Ç∞@Ç∞HÇ∞Xb;gradient_tape/model_5/conv2d_132/Conv2D/Conv2DBackpropInputh
π
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ç∞@Ç∞HÇ∞b1gradient_tape/model_5/conv2d_142/Conv2D/Cast/CasthuZUÖB
˝
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2ÄP8Ç∞@ÅòHÅòb/model_5/batch_normalization_56/FusedBatchNormV3hu  »B
~
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_nnﬁÄÄ ÄÄ*Ä2@8Å∞@Å∞HÅ∞Xbmodel_5/conv2d_138/Conv2Dh
±
Ìvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å∞@Å∞HÅ∞bmodel_5/dropout_30/dropout/CasthuZUÖB
¶	
Ávoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å∞@Å∞HÅ∞bmodel_5/activation_59/ReluhuZUÖB
ñ
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8Å∞@Å∞HÅ∞b<cond_1/then/_10/cond_1/Adam/Adam/update_42/ResourceApplyAdamhuZUÖB
–
Ùvoid cudnn::bn_fw_tr_1C11_singleread_specialized<__half2, 512, 1, 2, 0>(cudnnTensorStruct, __half2 const*, cudnnTensorStruct, __half2*, float const*, float const*, float, float, float*, float*, float*, float*, float, float, cudnn::reduced_divisor, int, cudnn::reduced_divisor, cudnn::bnFwPersistentState*, int, float, float, float, int, float, float, cudnnStatus_t*, bool)(ê ÄÄ*Ä2Ä8Ç®@Ç®HÇ®b/model_5/batch_normalization_57/FusedBatchNormV3hu  »B
¶	
Ávoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å®@Å®HÅ®bmodel_5/activation_60/ReluhuZUÖB
°
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_ntﬁÄÄ ÄÄ*Ä28Åê@ÅêHÅêXb<gradient_tape/model_5/conv2d_138/Conv2D/Conv2DBackpropFilterh
Ÿ

ü
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Åê@ÅêHÅêbAdam/gradients/AddN_4huZUÖB
≤
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å@ÅHÅbmodel_5/conv2d_142/Conv2D/CasthuZUÖB
ÿ
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2{8Å@ÅHÅb0gradient_tape/conv2d_143/kernel/Regularizer/TilehuZUÖB
ó
1ampere_s16816gemm_fp16_128x64_ldg8_stages_64x4_ntíÄÄ ÄÄ*Ä28Ç‡@Ç‡HÇ‡Xb<gradient_tape/model_5/conv2d_126/Conv2D/Conv2DBackpropFilterh
¶	
Ávoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ç‡@Ç‡HÇ‡bmodel_5/activation_62/ReluhuZUÖB
s
.ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_nnˆÄÄ*Ä2Ä8Ç∞@Ç∞HÇ∞Xbmodel_5/conv2d_126/Conv2DhugUÖA
†
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_tn‚ÄÄ ÄÄ*Ä2@8Å∞@Å∞HÅ∞Xb;gradient_tape/model_5/conv2d_138/Conv2D/Conv2DBackpropInputh

;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_nnﬁÄÄ ÄÄ*Ä2Ä8Å®@Å®HÅ®Xbmodel_5/conv2d_132/Conv2Dh
ï
.ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_tn˛ÄÄ*Ä2Ä8Å®@Å®HÅ®Xb;gradient_tape/model_5/conv2d_126/Conv2D/Conv2DBackpropInputhugUÖA
ü
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Å®@Å®HÅ®b4gradient_tape/model_5/conv2d_128/BiasAdd/BiasAddGradhuZUÖB
ü
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Å†@Å†HÅ†b4gradient_tape/model_5/conv2d_126/BiasAdd/BiasAddGradhuZUÖB
ü
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Å†@Å†HÅ†b4gradient_tape/model_5/conv2d_131/BiasAdd/BiasAddGradhuZUÖB
ü
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Åò@ÅòHÅòb4gradient_tape/model_5/conv2d_129/BiasAdd/BiasAddGradhuZUÖB
ü
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Åò@ÅòHÅòb4gradient_tape/model_5/conv2d_142/BiasAdd/BiasAddGradhuZUÖB
o
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä 8Åê@ÅêHÅêb,gradient_tape/model_5/dropout_30/dropout/MulhuZUÖB
o
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä 8Åê@ÅêHÅêb,gradient_tape/model_5/dropout_31/dropout/MulhuZUÖB
ü
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  28Åê@ÅêHÅêb4gradient_tape/model_5/conv2d_139/BiasAdd/BiasAddGradhuZUÖB
ü
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Åê@ÅêHÅêb4gradient_tape/model_5/conv2d_141/BiasAdd/BiasAddGradhuZUÖB
a
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä 8Çà@ÇàHÇàbmodel_5/dropout_30/dropout/MulhuZUÖB
a
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä 8ÅÄ@ÅÄHÅÄbmodel_5/dropout_31/dropout/MulhuZUÖB
ü
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  28ÅÄ@ÅÄHÅÄb4gradient_tape/model_5/conv2d_138/BiasAdd/BiasAddGradhuZUÖB
K
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2–8Å¯@Å¯HÅ¯bmul_54huZUÖB
Ö
√void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*Ä2{8Å¯@ÄàHÅ¯bmodel_5/concatenate_46/concathuZUÖB
t
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2–8Å@ÅHÅb/gradient_tape/conv2d_143/kernel/Regularizer/MulhuZUÖB
l
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2–8Å@ÅHÅb$conv2d_143/kernel/Regularizer/SquarehuZUÖB
ˇ
ívoid cudnn::bn_bw_1C11_singleread_specialized<__half2, 512, 1, 2, 7>(float, float, float, float, cudnnTensorStruct, __half2 const*, cudnnTensorStruct, __half2 const*, cudnnTensorStruct, __half2*, float const*, float*, float*, float const*, float const*, float, cudnn::reduced_divisor, int, cudnn::reduced_divisor, cudnn::bnBwPersistentState*, int, float, float, float, int, float, cudnnStatus_t*, bool)8ê Ä *Ä2Ä
8Å@ÅHÅbAgradient_tape/model_5/batch_normalization_56/FusedBatchNormGradV3huZUÖB
˙
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8ÅË@ÅËHÅËXb<gradient_tape/model_5/conv2d_143/Conv2D/Conv2DBackpropFilterhuZUÖB
π
htensorflow::functor::ReluGradHalfKernelVector(Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2Ä8Çÿ@ÇÿHÇÿb,gradient_tape/model_5/activation_61/ReluGradhu  »B
≠
Àvoid cutlass_cudnn::Kernel<cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4> >(cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4>::Params)6* 2@8Å»@Å»HÅ»Xb<gradient_tape/model_5/conv2d_136/Conv2D/Conv2DBackpropFilterhu  ñB
”
åvoid pooling_fw_kernel_max_nhwc<__half, float, 0, (cudnnNanPropagation_t)0, false>(PoolingParams, __half*, __half const*, float, float, int)*Ä2Ä†8Å∏@Å∏HÅ∏b model_5/max_pooling2d_22/MaxPoolhu  »B
π
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å∞@Å∞HÅ∞b1gradient_tape/model_5/conv2d_143/Conv2D/Cast/CasthuZUÖB
É
ßvoid tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>)*Ä2{8Å∞@Å∞HÅ∞b7model_5/dropout_31/dropout/random_uniform/RandomUniformhuZUÖB
T
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2ÄZ8Å®@Å®HÅ®bIsFinite_50hu  »B
s
'GreaterEqual_GPU_DT_HALF_DT_BOOL_kernel*Ä2Ä 8Å†@Å†HÅ†b'model_5/dropout_31/dropout/GreaterEqualhuZUÖB
≤
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Åò@ÅòHÅòbmodel_5/conv2d_143/Conv2D/CasthuZUÖB
ö
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Åê@ÅêHÅêb,gradient_tape/model_5/concatenate_45/Slice_2huZUÖB
ò
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Åê@ÅêHÅêb*gradient_tape/model_5/concatenate_47/SlicehuZUÖB
ö
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Åà@ÅàHÅàb,gradient_tape/model_5/concatenate_44/Slice_1huZUÖB
ö
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Åà@ÅàHÅàb,gradient_tape/model_5/concatenate_45/Slice_1huZUÖB
ö
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Åà@ÅàHÅàb,gradient_tape/model_5/concatenate_47/Slice_1huZUÖB
ö
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8ÅÄ@ÅÄHÅÄb,gradient_tape/model_5/concatenate_44/Slice_2huZUÖB
”
åvoid pooling_fw_kernel_max_nhwc<__half, float, 0, (cudnnNanPropagation_t)0, false>(PoolingParams, __half*, __half const*, float, float, int)*Ä2Äê8ÅÄ@ÅÄHÅÄb model_5/max_pooling2d_23/MaxPoolhu  »B
É
ßvoid tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>)*Ä2{8Å¯@Å¯HÅ¯b7model_5/dropout_30/dropout/random_uniform/RandomUniformhuZUÖB
t
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ä8ÅË@ÅËHÅËb/gradient_tape/dense_16/kernel/Regularizer/Mul_1huZUÖB
™
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8ÅË@ÅËHÅËbAdam/gradients/AddN_27huZUÖB
™
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8ÅË@ÅËHÅËbAdam/gradients/AddN_32huZUÖB
≠
Àvoid cutlass_cudnn::Kernel<cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4> >(cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4>::Params)6* 2@8ÅË@ÅËHÅËXb<gradient_tape/model_5/conv2d_133/Conv2D/Conv2DBackpropFilterhu  ñB
™
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8ÅË@ÅËHÅËbmodel_5/conv2d_142/BiasAddhuZUÖB
s
'GreaterEqual_GPU_DT_HALF_DT_BOOL_kernel*Ä2Ä 8Å‡@Å‡HÅ‡b'model_5/dropout_30/dropout/GreaterEqualhuZUÖB
v
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ë8Å‡@Å‡HÅ‡b1gradient_tape/conv2d_140/kernel/Regularizer/Mul_1huZUÖB
™
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å‡@Å‡HÅ‡bAdam/gradients/AddN_24huZUÖB
™
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Å‡@Å‡HÅ‡bmodel_5/conv2d_133/BiasAddhuZUÖB
™
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Å‡@Å‡HÅ‡bmodel_5/conv2d_137/BiasAddhuZUÖB
v
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ë8Ä‡@Ä‡HÄ‡b1gradient_tape/conv2d_137/kernel/Regularizer/Mul_1huZUÖB
™
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Ä‡@Ä‡HÄ‡bmodel_5/conv2d_134/BiasAddhuZUÖB
–
Ùvoid cudnn::bn_fw_tr_1C11_singleread_specialized<__half2, 512, 1, 2, 0>(cudnnTensorStruct, __half2 const*, cudnnTensorStruct, __half2*, float const*, float const*, float, float, float*, float*, float*, float*, float, float, cudnn::reduced_divisor, int, cudnn::reduced_divisor, cudnn::bnFwPersistentState*, int, float, float, float, int, float, float, cudnnStatus_t*, bool)(ê ÄÄ*Ä2Ä
8Åÿ@ÅÿHÅÿb/model_5/batch_normalization_56/FusedBatchNormV3hu  »B
™
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Åÿ@ÅÿHÅÿbmodel_5/conv2d_136/BiasAddhuZUÖB
™
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Äÿ@ÄÿHÄÿbmodel_5/conv2d_141/BiasAddhuZUÖB
ñ
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8Å–@Å–HÅ–b<cond_1/then/_10/cond_1/Adam/Adam/update_20/ResourceApplyAdamhuZUÖB
d
 ampere_fp16_sgemm_fp16_32x128_nn9ÄÄ*Ä2Ä 8Å»@Å»HÅ»Xbmodel_5/conv2d_123/Conv2DhuZUÖB
◊
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Å∏@Å∏HÅ∏Xbmodel_5/conv2d_137/Conv2DhuZUÖB
ƒ
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2ú8Å∞@Å∞HÅ∞b!conv2d_142/kernel/Regularizer/Sumhu  »B
˘
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Å®@Å®HÅ®Xb;gradient_tape/model_5/conv2d_137/Conv2D/Conv2DBackpropInputhuZUÖB
◊
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Å®@Å®HÅ®Xbmodel_5/conv2d_140/Conv2DhuZUÖB
¶	
Ávoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å†@Å†HÅ†bmodel_5/activation_61/ReluhuZUÖB
ç
§void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nt_align1>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nt_align1::Params)¢ Ä¿*Ä2¢8Å†@Å†HÅ†Xb<gradient_tape/model_5/conv2d_123/Conv2D/Conv2DBackpropFilterhugUÖA
ñ
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8Å†@Å†HÅ†b<cond_1/then/_10/cond_1/Adam/Adam/update_34/ResourceApplyAdamhuZUÖB
˘
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Å†@Å†HÅ†Xb;gradient_tape/model_5/conv2d_140/Conv2D/Conv2DBackpropInputhuZUÖB
Ÿ

ü
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Åê@ÅêHÅêbAdam/gradients/AddN_5huZUÖB
Æ
Nvoid cudnn::ops::scalePackedTensor_kernel<__half, float>(long, __half*, float)*Ä2ˇˇ8Äê@ÄêHÄêb:gradient_tape/model_5/max_pooling2d_22/MaxPool/MaxPoolGradhu  »B
T
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2¿>8Åà@ÅàHÅàbIsFinite_52hu  »B
™
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Åà@ÅàHÅàbAdam/gradients/AddN_21huZUÖB
ÿ
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2{8Åà@ÅàHÅàb0gradient_tape/conv2d_137/kernel/Regularizer/TilehuZUÖB
v
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8ÄÄ@ÄÄHÄÄb1gradient_tape/conv2d_134/kernel/Regularizer/Mul_1huZUÖB
ñ
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8Å¯@Å¯HÅ¯b<cond_1/then/_10/cond_1/Adam/Adam/update_48/ResourceApplyAdamhuZUÖB
Æ
Nvoid cudnn::ops::scalePackedTensor_kernel<__half, float>(long, __half*, float)*Ä2ˇˇ8Å@ÅHÅb:gradient_tape/model_5/max_pooling2d_23/MaxPool/MaxPoolGradhu  »B
ü
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8ÅË@ÅËHÅËb4gradient_tape/model_5/conv2d_133/BiasAdd/BiasAddGradhuZUÖB
ü
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8ÄË@ÄËHÄËb4gradient_tape/model_5/conv2d_136/BiasAdd/BiasAddGradhuZUÖB
v
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2–8Å‡@Å‡HÅ‡b1gradient_tape/conv2d_139/kernel/Regularizer/Mul_1huZUÖB
™
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å‡@Å‡HÅ‡bAdam/gradients/AddN_26huZUÖB
ü
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Å‡@Å‡HÅ‡b4gradient_tape/model_5/conv2d_134/BiasAdd/BiasAddGradhuZUÖB
ü
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Å‡@Å‡HÅ‡b4gradient_tape/model_5/conv2d_137/BiasAdd/BiasAddGradhuZUÖB
ñ
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8Äÿ@ÄÿHÄÿb<cond_1/then/_10/cond_1/Adam/Adam/update_18/ResourceApplyAdamhuZUÖB
˘
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Äÿ@ÄÿHÄÿXb;gradient_tape/model_5/conv2d_134/Conv2D/Conv2DBackpropInputhuZUÖB
j
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ä8Å–@Å–HÅ–b"dense_16/kernel/Regularizer/SquarehuZUÖB
ñ
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8Å–@Å–HÅ–b<cond_1/then/_10/cond_1/Adam/Adam/update_26/ResourceApplyAdamhuZUÖB
◊
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Å–@Å–HÅ–Xbmodel_5/conv2d_134/Conv2DhuZUÖB
r
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ä8Ä–@Ä–HÄ–b-gradient_tape/dense_16/kernel/Regularizer/MulhuZUÖB
t
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ë8Å»@Å»HÅ»b/gradient_tape/conv2d_137/kernel/Regularizer/MulhuZUÖB
t
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ë8Ä»@Ä»HÄ»b/gradient_tape/conv2d_140/kernel/Regularizer/MulhuZUÖB
K
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ä8Ä»@Ä»HÄ»bmul_62huZUÖB
K
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ë8Å¿@Å¿HÅ¿bmul_46huZUÖB
c
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä
8Å¿@Å¿HÅ¿b model_5/dropout_32/dropout/Mul_1huZUÖB
◊
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Å¿@Å¿HÅ¿Xbmodel_5/conv2d_139/Conv2DhuZUÖB
K
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ë8Ä¿@Ä¿HÄ¿bmul_38huZUÖB
l
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ë8Ä¿@Ä¿HÄ¿b$conv2d_137/kernel/Regularizer/SquarehuZUÖB
l
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ë8Ä¿@Ä¿HÄ¿b$conv2d_140/kernel/Regularizer/SquarehuZUÖB
ÿ
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2{8Ä¿@Ä¿HÄ¿b0gradient_tape/conv2d_134/kernel/Regularizer/TilehuZUÖB
˘
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ä¿@Ä¿HÄ¿Xb;gradient_tape/model_5/conv2d_139/Conv2D/Conv2DBackpropInputhuZUÖB
˙
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Å∏@Å∏HÅ∏Xb<gradient_tape/model_5/conv2d_140/Conv2D/Conv2DBackpropFilterhuZUÖB
q
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä
8Ä∏@Ä∏HÄ∏b.gradient_tape/model_5/dropout_32/dropout/Mul_1huZUÖB
˙
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ä∏@Ä∏HÄ∏Xb<gradient_tape/model_5/conv2d_137/Conv2D/Conv2DBackpropFilterhuZUÖB
ÿ
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2{8Å∞@Å∞HÅ∞b0gradient_tape/conv2d_139/kernel/Regularizer/TilehuZUÖB
ƒ
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2–8Ä∞@Ä∞HÄ∞b!conv2d_143/kernel/Regularizer/Sumhu  »B
∞
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä®@Ä®HÄ®bmodel_5/dense_16/MatMul/CasthuZUÖB
∑
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä®@Ä®HÄ®b/gradient_tape/model_5/dense_16/MatMul/Cast/CasthuZUÖB
q
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä	8Å†@Å†HÅ†b.gradient_tape/model_5/dropout_33/dropout/Mul_1huZUÖB
c
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä	8Å†@Å†HÅ†b model_5/dropout_33/dropout/Mul_1huZUÖB
π
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å†@Å†HÅ†b1gradient_tape/model_5/conv2d_137/Conv2D/Cast/CasthuZUÖB
π
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å†@Å†HÅ†b1gradient_tape/model_5/conv2d_140/Conv2D/Cast/CasthuZUÖB
≤
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Åò@ÅòHÅòbmodel_5/conv2d_137/Conv2D/CasthuZUÖB
ò
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Åò@ÅòHÅòb*gradient_tape/model_5/concatenate_45/SlicehuZUÖB
≤
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Äò@ÄòHÄòbmodel_5/conv2d_140/Conv2D/CasthuZUÖB
ò
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Äò@ÄòHÄòb*gradient_tape/model_5/concatenate_44/SlicehuZUÖB
ò
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Äò@ÄòHÄòb*gradient_tape/model_5/concatenate_46/SlicehuZUÖB
±
Ìvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Åê@ÅêHÅêbmodel_5/dropout_32/dropout/CasthuZUÖB
ö
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Åê@ÅêHÅêb,gradient_tape/model_5/concatenate_46/Slice_1huZUÖB
t
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8Åà@ÅàHÅàb/gradient_tape/conv2d_134/kernel/Regularizer/MulhuZUÖB
l
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8Åà@ÅàHÅàb$conv2d_134/kernel/Regularizer/SquarehuZUÖB
K
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8Äà@ÄàHÄàbmul_30huZUÖB
±
Ìvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8ÅÄ@ÅÄHÅÄbmodel_5/dropout_33/dropout/CasthuZUÖB
™
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8ÅÄ@ÅÄHÅÄbmodel_5/conv2d_132/BiasAddhuZUÖB
˙
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8ÄÄ@ÄÄHÄÄXb<gradient_tape/model_5/conv2d_134/Conv2D/Conv2DBackpropFilterhuZUÖB
™
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Å¯@Å¯HÅ¯bmodel_5/conv2d_135/BiasAddhuZUÖB
t
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2–8Ä¯@Ä¯HÄ¯b/gradient_tape/conv2d_139/kernel/Regularizer/MulhuZUÖB
™
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Ä¯@Ä¯HÄ¯bmodel_5/conv2d_138/BiasAddhuZUÖB
™
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Ä¯@Ä¯HÄ¯bmodel_5/conv2d_139/BiasAddhuZUÖB
K
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2–8Å@ÅHÅbmul_44huZUÖB
π
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å@ÅHÅb1gradient_tape/model_5/conv2d_134/Conv2D/Cast/CasthuZUÖB
l
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2–8Ä@ÄHÄb$conv2d_139/kernel/Regularizer/SquarehuZUÖB
˙
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ä@ÄHÄXb<gradient_tape/model_5/conv2d_139/Conv2D/Conv2DBackpropFilterhuZUÖB
≤
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8ÄË@ÄËHÄËbmodel_5/conv2d_134/Conv2D/CasthuZUÖB
T
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2Ä 8Åÿ@ÅÿHÅÿbIsFinite_60hu  »B
o
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä
8Äÿ@ÄÿHÄÿb,gradient_tape/model_5/dropout_32/dropout/MulhuZUÖB
≤
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å–@Å–HÅ–bmodel_5/conv2d_139/Conv2D/CasthuZUÖB
T
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2†8Ä–@Ä–HÄ–bIsFinite_44hu  »B
a
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä
8Ä–@Ä–HÄ–bmodel_5/dropout_32/dropout/MulhuZUÖB
π
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä–@Ä–HÄ–b1gradient_tape/model_5/conv2d_139/Conv2D/Cast/CasthuZUÖB
™
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä–@Ä–HÄ–bAdam/gradients/AddN_18huZUÖB
T
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2†8Å»@Å»HÅ»bIsFinite_36hu  »B
v
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2ê8Å»@Å»HÅ»b1gradient_tape/conv2d_131/kernel/Regularizer/Mul_1huZUÖB
÷
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Å»@Å»HÅ»b.gradient_tape/dense_16/kernel/Regularizer/TilehuZUÖB
o
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä	8Ä»@Ä»HÄ»b,gradient_tape/model_5/dropout_33/dropout/MulhuZUÖB
a
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä	8Ä¿@Ä¿HÄ¿bmodel_5/dropout_33/dropout/MulhuZUÖB
¬
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2Ä8Ä¿@Ä¿HÄ¿bdense_16/kernel/Regularizer/Sumhu  »B
ü
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Ä¿@Ä¿HÄ¿b4gradient_tape/model_5/conv2d_135/BiasAdd/BiasAddGradhuZUÖB
˘
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ä¿@Ä¿HÄ¿Xb;gradient_tape/model_5/conv2d_131/Conv2D/Conv2DBackpropInputhuZUÖB
™
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å∏@Å∏HÅ∏bAdam/gradients/AddN_23huZUÖB
ü
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Å∏@Å∏HÅ∏b4gradient_tape/model_5/conv2d_132/BiasAdd/BiasAddGradhuZUÖB
◊
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Å∏@Å∏HÅ∏Xbmodel_5/conv2d_131/Conv2DhuZUÖB
v
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ë8Ä∏@Ä∏HÄ∏b1gradient_tape/conv2d_136/kernel/Regularizer/Mul_1huZUÖB
ƒ
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2Ë8Å∞@Å∞HÅ∞b!conv2d_137/kernel/Regularizer/Sumhu  »B
ƒ
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2Ë8Å∞@Å∞HÅ∞b!conv2d_140/kernel/Regularizer/Sumhu  »B
Ñ
Campere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x5_nníÄÄ ÄÄ*Ä2 8Ä∞@Ä∞HÄ∞Xbmodel_5/dense_16/MatMulh
í
Campere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tníÄÄ ÄÄ*Ä2 8Ä∞@Ä∞HÄ∞Xb%gradient_tape/model_5/dense_16/MatMulh
™
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å®@Å®HÅ®bAdam/gradients/AddN_28huZUÖB
ÿ
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2{8Å®@Å®HÅ®b0gradient_tape/conv2d_131/kernel/Regularizer/TilehuZUÖB
É
ßvoid tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>)*Ä2{8Å®@Å®HÅ®b7model_5/dropout_32/dropout/random_uniform/RandomUniformhuZUÖB
◊
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Å®@Å®HÅ®Xbmodel_5/conv2d_136/Conv2DhuZUÖB
s
'GreaterEqual_GPU_DT_HALF_DT_BOOL_kernel*Ä2Ä
8Ä®@Ä®HÄ®b'model_5/dropout_32/dropout/GreaterEqualhuZUÖB
T
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2Ä8Ä®@Ä®HÄ®bIsFinite_28hu  »B
˘
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ä®@Ä®HÄ®Xb;gradient_tape/model_5/conv2d_136/Conv2D/Conv2DBackpropInputhuZUÖB
ÿ
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2{8Å†@Å†HÅ†b0gradient_tape/conv2d_136/kernel/Regularizer/TilehuZUÖB
›
˙void nhwcAddPaddingKernel<__half, __half, float, true, (cudnnKernelDataType_t)0>(int, int, int, int, int, int, int, int, __half const*, __half*, int, int, int, int, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor)*Ä2§8Å†@ÄHÅàXb<gradient_tape/model_5/conv2d_124/Conv2D/Conv2DBackpropFilterhu  »B
v
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2¿8Ä†@Ä†HÄ†b1gradient_tape/conv2d_141/kernel/Regularizer/Mul_1huZUÖB
É
ßvoid tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>)*Ä2{8Ä†@Ä†HÄ†b7model_5/dropout_33/dropout/random_uniform/RandomUniformhuZUÖB
ö
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Åò@ÅòHÅòb,gradient_tape/model_5/concatenate_47/Slice_2huZUÖB
T
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2¿8Äò@ÄòHÄòbIsFinite_42hu  »B
v
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8Äò@ÄòHÄòb1gradient_tape/conv2d_133/kernel/Regularizer/Mul_1huZUÖB
™
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Äò@ÄòHÄòbAdam/gradients/AddN_17huZUÖB
™
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Äò@ÄòHÄòbAdam/gradients/AddN_20huZUÖB
›
˙void nhwcAddPaddingKernel<__half, __half, float, true, (cudnnKernelDataType_t)0>(int, int, int, int, int, int, int, int, __half const*, __half*, int, int, int, int, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor)*Ä2§8Äò@ÄHÄàXb<gradient_tape/model_5/conv2d_125/Conv2D/Conv2DBackpropFilterhu  »B
˙
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Äò@ÄòHÄòXb<gradient_tape/model_5/conv2d_131/Conv2D/Conv2DBackpropFilterhuZUÖB
v
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8Åê@ÅêHÅêb1gradient_tape/conv2d_130/kernel/Regularizer/Mul_1huZUÖB
s
'GreaterEqual_GPU_DT_HALF_DT_BOOL_kernel*Ä2Ä	8Äê@ÄêHÄêb'model_5/dropout_33/dropout/GreaterEqualhuZUÖB
t
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2ê8Äê@ÄêHÄêb/gradient_tape/conv2d_131/kernel/Regularizer/MulhuZUÖB
ö
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Äê@ÄêHÄêb,gradient_tape/model_5/concatenate_46/Slice_2huZUÖB
ÿ
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2{8Äê@ÄêHÄêb0gradient_tape/conv2d_130/kernel/Regularizer/TilehuZUÖB
ÿ
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2{8Äê@ÄêHÄêb0gradient_tape/conv2d_141/kernel/Regularizer/TilehuZUÖB
å
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2†8Äê@ÄêHÄêbAll_50hu  »B
˘
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Äê@ÄêHÄêXb;gradient_tape/model_5/conv2d_130/Conv2D/Conv2DBackpropInputhuZUÖB
˘
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Äê@ÄêHÄêXb;gradient_tape/model_5/conv2d_133/Conv2D/Conv2DBackpropInputhuZUÖB
◊
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Äê@ÄêHÄêXbmodel_5/conv2d_130/Conv2DhuZUÖB
◊
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Äê@ÄêHÄêXbmodel_5/conv2d_133/Conv2DhuZUÖB
™
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Åà@ÅàHÅàbmodel_5/conv2d_140/BiasAddhuZUÖB
K
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2ê8Äà@ÄàHÄàbmul_22huZUÖB
l
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2ê8Äà@ÄàHÄàb$conv2d_131/kernel/Regularizer/SquarehuZUÖB
ƒ
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2†8Äà@ÄàHÄàb!conv2d_134/kernel/Regularizer/Sumhu  »B
™
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Äà@ÄàHÄàbmodel_5/conv2d_143/BiasAddhuZUÖB
≤
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8ÅÄ@ÅÄHÅÄbmodel_5/conv2d_131/Conv2D/CasthuZUÖB
ñ
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8ÅÄ@ÅÄHÅÄb<cond_1/then/_10/cond_1/Adam/Adam/update_12/ResourceApplyAdamhuZUÖB
t
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ë8ÄÄ@ÄÄHÄÄb/gradient_tape/conv2d_136/kernel/Regularizer/MulhuZUÖB
K
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ë8ÄÄ@ÄÄHÄÄbmul_36huZUÖB
´
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8ÄÄ@ÄÄHÄÄbmodel_5/conv2d_123/CasthuZUÖB
π
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8ÄÄ@ÄÄHÄÄb1gradient_tape/model_5/conv2d_131/Conv2D/Cast/CasthuZUÖB
ƒ
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2–8ÄÄ@ÄÄHÄÄb!conv2d_139/kernel/Regularizer/Sumhu  »B
ü
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8ÄÄ@ÄÄHÄÄb4gradient_tape/model_5/conv2d_140/BiasAdd/BiasAddGradhuZUÖB
H
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2¿8Åx@ÅxHÅxbmul_50huZUÖB
i
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ë8Åx@ÅxHÅxb$conv2d_136/kernel/Regularizer/SquarehuZUÖB
ú
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Åx@ÅxHÅxb4gradient_tape/model_5/conv2d_143/BiasAdd/BiasAddGradhuZUÖB
˜
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Åx@ÅxHÅxXb<gradient_tape/model_5/conv2d_136/Conv2D/Conv2DBackpropFilterhuZUÖB
q
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2¿8Äx@ÄxHÄxb/gradient_tape/conv2d_141/kernel/Regularizer/MulhuZUÖB
ï
Ávoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<unsigned char const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<unsigned char const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Äx@ÄxHÄxbmodel_5/CasthuZUÖB
Ã
Êvoid xmma_cudnn::gemm::split_k_kernel<xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, xmma_cudnn::Col, 256, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 3> >(xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, xmma_cudnn::Col, 256, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 3>::Params)? Äê*Ä2	8Äx@ÄxHÄxPXb<gradient_tape/model_5/conv2d_127/Conv2D/Conv2DBackpropFilterhuMUB
â
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2Ë8Åp@ÅpHÅpbAll_52hu  »B
q
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8Äp@ÄpHÄpb/gradient_tape/conv2d_133/kernel/Regularizer/MulhuZUÖB
i
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2¿8Äp@ÄpHÄpb$conv2d_141/kernel/Regularizer/SquarehuZUÖB
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Äp@ÄpHÄpbmodel_5/conv2d_136/Conv2D/CasthuZUÖB
Ì
ëvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorForcedEvalOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorForcedEvalOp<Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_log_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorForcedEvalOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorForcedEvalOp<Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_log_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long)2*Ä28Äp@ÄpHÄpb:categorical_crossentropy/softmax_cross_entropy_with_logitshuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8Äp@ÄpHÄpb<cond_1/then/_10/cond_1/Adam/Adam/update_10/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8Äp@ÄpHÄpb<cond_1/then/_10/cond_1/Adam/Adam/update_40/ResourceApplyAdamhuZUÖB
H
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8Åh@ÅhHÅhbmul_28huZUÖB
q
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8Äh@ÄhHÄhb/gradient_tape/conv2d_130/kernel/Regularizer/MulhuZUÖB
H
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8Äh@ÄhHÄhbmul_20huZUÖB
á
:ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_ntÍÄÄ*Ä28Äh@ÄhHÄhb'gradient_tape/model_5/dense_16/MatMul_1hugUÖA
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Äh@ÄhHÄhbmodel_5/conv2d_141/Conv2D/CasthuZUÖB
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Äh@ÄhHÄhb1gradient_tape/model_5/conv2d_136/Conv2D/Cast/CasthuZUÖB
˜
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Äh@ÄhHÄhXb<gradient_tape/model_5/conv2d_130/Conv2D/Conv2DBackpropFilterhuZUÖB
Q
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2¿8Å`@Å`HÅ`bIsFinite_20hu  »B
i
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8Å`@Å`HÅ`b$conv2d_130/kernel/Regularizer/SquarehuZUÖB
i
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8Å`@Å`HÅ`b$conv2d_133/kernel/Regularizer/SquarehuZUÖB
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä`@Ä`HÄ`bmodel_5/conv2d_130/Conv2D/CasthuZUÖB
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä`@Ä`HÄ`bmodel_5/conv2d_133/Conv2D/CasthuZUÖB
˜
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ä`@Ä`HÄ`Xb<gradient_tape/model_5/conv2d_133/Conv2D/Conv2DBackpropFilterhuZUÖB
Q
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2Ä
8ÅX@ÅXHÅXbIsFinite_48hu  »B
Q
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2†8ÄX@ÄXHÄXbIsFinite_34hu  »B
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8ÄX@ÄXHÄXb1gradient_tape/model_5/conv2d_141/Conv2D/Cast/CasthuZUÖB
¡
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2Ë8ÄX@ÄXHÄXb!conv2d_136/kernel/Regularizer/Sumhu  »B
¡
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2ê8ÄX@ÄXHÄXb!conv2d_131/kernel/Regularizer/Sumhu  »B
Q
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2Ä	8ÄP@ÄPHÄPbIsFinite_18hu  »B
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8ÄP@ÄPHÄPb1gradient_tape/model_5/conv2d_130/Conv2D/Cast/CasthuZUÖB
Q
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2Ä	8ÄH@ÄHHÄHbIsFinite_26hu  »B
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8ÄH@ÄHHÄHb1gradient_tape/model_5/conv2d_133/Conv2D/Cast/CasthuZUÖB
’
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2{8ÄH@ÄHHÄHb0gradient_tape/conv2d_128/kernel/Regularizer/TilehuZUÖB
â
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2Ù8ÄH@ÄHHÄHbAll_44hu  »B
¡
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2†8ÄH@ÄHHÄHb!conv2d_130/kernel/Regularizer/Sumhu  »B
ˆ
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8ÄH@ÄHHÄHXb;gradient_tape/model_5/conv2d_128/Conv2D/Conv2DBackpropInputhuZUÖB
‘
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8ÄH@ÄHHÄHXbmodel_5/conv2d_128/Conv2DhuZUÖB
Î
èvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorReshapingOp<Eigen::IndexList<int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> >, Eigen::TensorReductionOp<Eigen::internal::MaxReducer<float, 0>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorReshapingOp<Eigen::IndexList<int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> >, Eigen::TensorReductionOp<Eigen::internal::MaxReducer<float, 0>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long)@*Ä28Å@@Å@HÅ@b:categorical_crossentropy/softmax_cross_entropy_with_logitshuZUÖB
r
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2d8Ä@@Ä@HÄ@b1gradient_tape/conv2d_128/kernel/Regularizer/Mul_1huZUÖB
ß
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä@@Ä@HÄ@bAdam/gradients/AddN_15huZUÖB
â
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2ê8Ä@@Ä@HÄ@bAll_28hu  »B
â
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2Ù8Ä@@Ä@HÄ@bAll_36hu  »B
¡
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2¿8Ä@@Ä@HÄ@b!conv2d_141/kernel/Regularizer/Sumhu  »B
r
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2H8Å8@Å8HÅ8b1gradient_tape/conv2d_127/kernel/Regularizer/Mul_1huZUÖB
r
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2P8Ä8@Ä8HÄ8b1gradient_tape/conv2d_138/kernel/Regularizer/Mul_1huZUÖB
ß
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä8@Ä8HÄ8bAdam/gradients/AddN_14huZUÖB
ß
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä8@Ä8HÄ8bAdam/gradients/AddN_25huZUÖB
Â
âvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorReshapingOp<Eigen::IndexList<int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> >, Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorReshapingOp<Eigen::IndexList<int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> >, Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long)/*Ä28Ä8@Ä8HÄ8b:categorical_crossentropy/softmax_cross_entropy_with_logitshuZUÖB
â
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2Ë8Ä8@Ä8HÄ8bAll_42hu  »B
â
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2Ä8Ä8@Ä8HÄ8bAll_60hu  »B
¡
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2†8Ä8@Ä8HÄ8b!conv2d_133/kernel/Regularizer/Sumhu  »B
™
Àvoid cutlass_cudnn::Kernel<cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4> >(cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4>::Params)6* 28Ä8@Ä8HÄ8Xb<gradient_tape/model_5/conv2d_125/Conv2D/Conv2DBackpropFilterhu  ñB
ˆ
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ä8@Ä8HÄ8Xb;gradient_tape/model_5/conv2d_127/Conv2D/Conv2DBackpropInputhuZUÖB
‘
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ä8@Ä8HÄ8Xbmodel_5/conv2d_127/Conv2DhuZUÖB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2P8Å0@Å0HÅ0b/gradient_tape/conv2d_138/kernel/Regularizer/MulhuZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2d8Å0@Å0HÅ0bmul_14huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2P8Ä0@Ä0HÄ0bmul_42huZUÖB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2d8Ä0@Ä0HÄ0b/gradient_tape/conv2d_128/kernel/Regularizer/MulhuZUÖB
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä0@Ä0HÄ0b1gradient_tape/model_5/conv2d_128/Conv2D/Cast/CasthuZUÖB
’
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä28Ä0@Ä0HÄ0b0gradient_tape/conv2d_125/kernel/Regularizer/TilehuZUÖB
’
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2@8Ä0@Ä0HÄ0b0gradient_tape/conv2d_129/kernel/Regularizer/TilehuZUÖB
’
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2{8Ä0@Ä0HÄ0b0gradient_tape/conv2d_127/kernel/Regularizer/TilehuZUÖB
’
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2{8Ä0@Ä0HÄ0b0gradient_tape/conv2d_138/kernel/Regularizer/TilehuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2@8Ä0@Ä0HÄ0b<cond_1/then/_10/cond_1/Adam/Adam/update_16/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2P8Ä0@Ä0HÄ0b<cond_1/then/_10/cond_1/Adam/Adam/update_32/ResourceApplyAdamhuZUÖB
ä
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Ä0@Ä0HÄ0b4gradient_tape/model_5/conv2d_134/BiasAdd/BiasAddGradhuZUÖB
˜
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ä0@Ä0HÄ0Xb<gradient_tape/model_5/conv2d_128/Conv2D/Conv2DBackpropFilterhuZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2H8Å(@Å(HÅ(bmul_12huZUÖB
Q
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2†8Ä(@Ä(HÄ(bIsFinite_10hu  »B
Q
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2¿8Ä(@Ä(HÄ(bIsFinite_40hu  »B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2H8Ä(@Ä(HÄ(b/gradient_tape/conv2d_127/kernel/Regularizer/MulhuZUÖB
h
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2H8Ä(@Ä(HÄ(b$conv2d_127/kernel/Regularizer/SquarehuZUÖB
h
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2P8Ä(@Ä(HÄ(b$conv2d_138/kernel/Regularizer/SquarehuZUÖB
h
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2d8Ä(@Ä(HÄ(b$conv2d_128/kernel/Regularizer/SquarehuZUÖB
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä(@Ä(HÄ(b1gradient_tape/model_5/conv2d_127/Conv2D/Cast/CasthuZUÖB
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä(@Ä(HÄ(b1gradient_tape/model_5/conv2d_138/Conv2D/Cast/CasthuZUÖB
’
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä28Ä(@Ä(HÄ(b0gradient_tape/conv2d_126/kernel/Regularizer/TilehuZUÖB
’
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2P8Ä(@Ä(HÄ(b0gradient_tape/conv2d_135/kernel/Regularizer/TilehuZUÖB
â
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2ê8Ä(@Ä(HÄ(bAll_18hu  »B
â
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2†8Ä(@Ä(HÄ(bAll_48hu  »B
â
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2¥8Ä(@Ä(HÄ(bAll_34hu  »B
â
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2»8Ä(@Ä(HÄ(bAll_20hu  »B
¿
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2d8Ä(@Ä(HÄ(b!conv2d_128/kernel/Regularizer/Sumhu  »B
„
§void cutlass::Kernel<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2>(cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2::Params)_ Ä§*Ä28Ä(@Ä(HÄ(Xbmodel_5/dense_17/MatMulhugUÖA
™
Àvoid cutlass_cudnn::Kernel<cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4> >(cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4>::Params)6* 28Ä(@Ä(HÄ(Xb<gradient_tape/model_5/conv2d_124/Conv2D/Conv2DBackpropFilterhu  ñB
Ñ
§void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*) *Ä28Ä(@Ä(HÄ(Xb<gradient_tape/model_5/conv2d_123/Conv2D/Conv2DBackpropFilterhu  »B
Ö
§void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*) *Ä2Ä8Ä(@Ä(HÄ(Xb<gradient_tape/model_5/conv2d_138/Conv2D/Conv2DBackpropFilterhu  »B
Ë
©void tensorflow::(anonymous namespace)::GenerateNormalizedProb<Eigen::half, float, 8>(Eigen::half const*, float const*, Eigen::half const*, Eigen::half*, int, int, bool)*Ä28Ä(@Ä(HÄ(bmodel_5/activation_65/Softmaxhu  »B
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä(@Ä(HÄ(b<cond_1/then/_10/cond_1/Adam/Adam/update_65/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2@8Ä(@Ä(HÄ(b<cond_1/then/_10/cond_1/Adam/Adam/update_24/ResourceApplyAdamhuZUÖB
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä(@Ä(HÄ(bAll_54hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä2 8Ä(@Ä(HÄ(bAll_16hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä2 8Ä(@Ä(HÄ(bAll_24hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä2 8Ä(@Ä(HÄ(bAll_32hu  »B
ä
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Ä(@Ä(HÄ(b4gradient_tape/model_5/conv2d_127/BiasAdd/BiasAddGradhuZUÖB
ä
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Ä(@Ä(HÄ(b4gradient_tape/model_5/conv2d_130/BiasAdd/BiasAddGradhuZUÖB
ä
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Ä(@Ä(HÄ(b4gradient_tape/model_5/conv2d_133/BiasAdd/BiasAddGradhuZUÖB
ä
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Ä(@Ä(HÄ(b4gradient_tape/model_5/conv2d_136/BiasAdd/BiasAddGradhuZUÖB
ä
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Ä(@Ä(HÄ(b4gradient_tape/model_5/conv2d_137/BiasAdd/BiasAddGradhuZUÖB
ä
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Ä(@Ä(HÄ(b4gradient_tape/model_5/conv2d_140/BiasAdd/BiasAddGradhuZUÖB
ä
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Ä(@Ä(HÄ(b4gradient_tape/model_5/conv2d_143/BiasAdd/BiasAddGradhuZUÖB
˜
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ä(@Ä(HÄ(Xb<gradient_tape/model_5/conv2d_127/Conv2D/Conv2DBackpropFilterhuZUÖB
ß
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Å @Å HÅ bAdam/gradients/AddN_16huZUÖB
ˆ	
ø	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Å @Å HÅ bAdam/gradients/AddN_1huZUÖB
≈
Èvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_quotient_op<float, float>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_quotient_op<float, float>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long).*Ä28Å @Å HÅ b:categorical_crossentropy/softmax_cross_entropy_with_logitshuZUÖB
¿
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2P8Å @Å HÅ b!conv2d_138/kernel/Regularizer/Sumhu  »B
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Å @Å HÅ b<cond_1/then/_10/cond_1/Adam/Adam/update_19/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Å @Å HÅ b<cond_1/then/_10/cond_1/Adam/Adam/update_43/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Å @Å HÅ b<cond_1/then/_10/cond_1/Adam/Adam/update_59/ResourceApplyAdamhuZUÖB
í
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Å @Å HÅ b;cond_1/then/_10/cond_1/Adam/Adam/update_4/ResourceApplyAdamhuZUÖB
ä
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Å @Å HÅ b4gradient_tape/model_5/conv2d_129/BiasAdd/BiasAddGradhuZUÖB
Q
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2ê8Ä @Ä HÄ bIsFinite_12hu  »B
r
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä @Ä HÄ b1gradient_tape/conv2d_129/kernel/Regularizer/Mul_1huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä @Ä HÄ bmul_34huZUÖB
h
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä @Ä HÄ b$conv2d_129/kernel/Regularizer/SquarehuZUÖB
l
 ampere_fp16_sgemm_fp16_128x32_tn9ÄÄ*Ä28Ä @Ä HÄ Xb%gradient_tape/model_5/dense_17/MatMulhuZUÖB
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä @Ä HÄ bmodel_5/conv2d_128/Conv2D/CasthuZUÖB
ß
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä @Ä HÄ bAdam/gradients/AddN_13huZUÖB
ß
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä @Ä HÄ bAdam/gradients/AddN_19huZUÖB
›
ìvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long) *Ä28Ä @ÄHÄb(ArithmeticOptimizer/AddOpsRewrite_AddN_1huZUÖB
”
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä28Ä @Ä HÄ b.gradient_tape/dense_17/kernel/Regularizer/TilehuZUÖB
¡	
Ávoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_squared_difference_op<float>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_squared_difference_op<float>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, int) *Ä2@8Ä @Ä HÄ b8model_5/batch_normalization_58/moments/SquaredDifferencehuZUÖB
≠
—void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorEvalToOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorForcedEvalOp<Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_log_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorEvalToOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorForcedEvalOp<Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_log_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>, long)(*Ä28Ä @Ä HÄ b:categorical_crossentropy/softmax_cross_entropy_with_logitshuZUÖB
à
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2(8Ä @Ä HÄ bAll_40hu  »B
à
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä228Ä @Ä HÄ bAll_12hu  »B
â
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2ê8Ä @Ä HÄ bAll_26hu  »B
¿
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2H8Ä @Ä HÄ b!conv2d_127/kernel/Regularizer/Sumhu  »B
æ
˘void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float>, float>(float*, float*, int, tensorflow::functor::Sum<float>, float)0*Ä28Ä @Ä HÄ b!conv2d_142/kernel/Regularizer/Sumhu  »B
º
˘void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float>, float>(float*, float*, int, tensorflow::functor::Sum<float>, float)0*Ä28Ä @Ä HÄ bdense_15/kernel/Regularizer/Sumhu  »B
Ò
§void cutlass::Kernel<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nt_align2>(cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nt_align2::Params)_ Ä¿*Ä2@8Ä @Ä HÄ b'gradient_tape/model_5/dense_17/MatMul_1hugUÖA
Ö
§void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*) *Ä2Ä8Ä @Ä HÄ Xb<gradient_tape/model_5/conv2d_129/Conv2D/Conv2DBackpropFilterhu  »B
‡
§void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*) *Ä2Ä8Ä @Ä HÄ Xbmodel_5/dense_15/MatMulhu  »B
É
¢void splitKreduce_kernel<float, __half, float, __half>(cublasSplitKParams<float>, float const*, __half const*, __half*, float const*, float const*, __half const*) *Ä2Ä8Ä @Ä HÄ Xb<gradient_tape/model_5/conv2d_126/Conv2D/Conv2DBackpropFilterhu  »B
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_11/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_13/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_15/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_17/ResourceApplyAdamhuZUÖB
í
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b;cond_1/then/_10/cond_1/Adam/Adam/update_2/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_21/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_22/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_23/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_25/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_29/ResourceApplyAdamhuZUÖB
í
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b;cond_1/then/_10/cond_1/Adam/Adam/update_3/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_30/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_31/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_33/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_35/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_37/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_39/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_41/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_45/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_49/ResourceApplyAdamhuZUÖB
í
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b;cond_1/then/_10/cond_1/Adam/Adam/update_5/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_51/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_53/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_47/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_57/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_58/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_61/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_62/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_63/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_54/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_55/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_64/ResourceApplyAdamhuZUÖB
í
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b;cond_1/then/_10/cond_1/Adam/Adam/update_8/ResourceApplyAdamhuZUÖB
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä @Ä HÄ bAll_46hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä @Ä HÄ bAll_55hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä @Ä HÄ bAll_57hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä @Ä HÄ bAll_58hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä @Ä HÄ bAll_61hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä @Ä HÄ bAll_62hu  »B
◊
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä2 8Ä @Ä HÄ bAll_8hu  »B
á
¬void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*Ä28Ä @Ä HÄ b!conv2d_124/kernel/Regularizer/Sumhu  »B
á
¬void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*Ä2 8Ä @Ä HÄ b!conv2d_129/kernel/Regularizer/Sumhu  »B
ä
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Ä @Ä HÄ b4gradient_tape/model_5/conv2d_126/BiasAdd/BiasAddGradhuZUÖB
ä
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Ä @Ä HÄ b4gradient_tape/model_5/conv2d_131/BiasAdd/BiasAddGradhuZUÖB
ä
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Ä @Ä HÄ b4gradient_tape/model_5/conv2d_132/BiasAdd/BiasAddGradhuZUÖB
ä
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Ä @Ä HÄ b4gradient_tape/model_5/conv2d_135/BiasAdd/BiasAddGradhuZUÖB
©
√void tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)Ä!*  2@8Ä @Ä HÄ bBgradient_tape/model_5/batch_normalization_58/batchnorm/add_1/Sum_1huZUÖB
Ä
ßvoid tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>)*Ä2@8Ä @Ä HÄ b7model_5/dropout_34/dropout/random_uniform/RandomUniformhuZUÖB
Ä
ßvoid tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>)*Ä2@8Ä @Ä HÄ b7model_5/dropout_35/dropout/random_uniform/RandomUniformhuZUÖB
q
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2@8Å@ÅHÅb.model_5/batch_normalization_59/batchnorm/add_1huZUÖB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Å@ÅHÅb!conv2d_139/kernel/Regularizer/mulhuZUÖB
m
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Å@ÅHÅb,model_5/batch_normalization_58/batchnorm/mulhuZUÖB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Å@ÅHÅb/gradient_tape/conv2d_125/kernel/Regularizer/MulhuZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Å@ÅHÅbmul_18huZUÖB
Å
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2@8Å@ÅHÅb@gradient_tape/model_5/batch_normalization_59/batchnorm/mul_1/MulhuZUÖB
µ
htensorflow::functor::ReluGradHalfKernelVector(Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä28Å@ÅHÅb,gradient_tape/model_5/activation_63/ReluGradhu  »B
ï
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Å@ÅHÅbCasthuZUÖB
∑
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Å@ÅHÅb2gradient_tape/model_5/conv2d_132/BiasAdd/Cast/CasthuZUÖB
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Å@ÅHÅb1gradient_tape/model_5/conv2d_129/Conv2D/Cast/CasthuZUÖB
Â
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2@8Å@ÅHÅb@gradient_tape/model_5/batch_normalization_59/moments/BroadcastTohuZUÖB
’
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä28Å@ÅHÅb0gradient_tape/conv2d_124/kernel/Regularizer/TilehuZUÖB
à
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2$8Å@ÅHÅbAll_10hu  »B
º
˘void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float>, float>(float*, float*, int, tensorflow::functor::Sum<float>, float)0*Ä28Å@ÅHÅbdense_16/kernel/Regularizer/Sumhu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Å@ÅHÅbAll_14hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Å@ÅHÅbAll_29hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Å@ÅHÅbAll_33hu  »B
o
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb,model_5/batch_normalization_58/batchnorm/addhuZUÖB
q
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2@8Ä@ÄHÄb.model_5/batch_normalization_58/batchnorm/add_1huZUÖB
X
"AddV2_GPU_DT_INT64_DT_INT64_kernel*Ä28Ä@ÄHÄbcond/then/_0/cond/addhuZUÖB
b
"AddV2_GPU_DT_INT64_DT_INT64_kernel*Ä28Ä@ÄHÄbcond_1/then/_10/cond_1/Adam/addhuZUÖB

 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2@8Ä@ÄHÄb>gradient_tape/model_5/batch_normalization_59/moments/truediv_1huZUÖB
o
'GreaterEqual_GPU_DT_HALF_DT_BOOL_kernel*Ä28Ä@ÄHÄb'model_5/dropout_34/dropout/GreaterEqualhuZUÖB
g
(GreaterEqual_GPU_DT_INT64_DT_BOOL_kernel*Ä28Ä@ÄHÄbcond/then/_0/cond/GreaterEqualhuZUÖB
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_11hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_15hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_19hu  »B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄb
IsFinite_2hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_22hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_29hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_31hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_35hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_37hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_39hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_43hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_53hu  »B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄb
IsFinite_6hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_46hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_59hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_63hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_54hu  »B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄb
IsFinite_8hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2@8Ä@ÄHÄbIsFinite_16hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2@8Ä@ÄHÄbIsFinite_24hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2P8Ä@ÄHÄbIsFinite_32hu  »B
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb!conv2d_127/kernel/Regularizer/mulhuZUÖB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb!conv2d_128/kernel/Regularizer/mulhuZUÖB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb!conv2d_133/kernel/Regularizer/mulhuZUÖB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb!conv2d_134/kernel/Regularizer/mulhuZUÖB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb!conv2d_136/kernel/Regularizer/mulhuZUÖB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb/gradient_tape/conv2d_123/kernel/Regularizer/MulhuZUÖB
Å
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb@gradient_tape/model_5/batch_normalization_58/batchnorm/mul/Mul_1huZUÖB
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb4model_5/batch_normalization_59/AssignMovingAvg_1/mulhuZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_15huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_27huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_32huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_35huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_37huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_39huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_48huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_49huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_51huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_53huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_56huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_57huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_63huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_65huZUÖB
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_9huZUÖB
r
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb1gradient_tape/conv2d_124/kernel/Regularizer/Mul_1huZUÖB
Å
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb@gradient_tape/model_5/batch_normalization_58/batchnorm/mul_2/MulhuZUÖB
Å
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb@gradient_tape/model_5/batch_normalization_59/batchnorm/mul_2/MulhuZUÖB
É
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbBgradient_tape/model_5/batch_normalization_59/batchnorm/mul_2/Mul_1huZUÖB
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb.model_5/batch_normalization_58/batchnorm/mul_2huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_60huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_64huZUÖB
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_6huZUÖB
r
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb1gradient_tape/conv2d_125/kernel/Regularizer/Mul_1huZUÖB
n
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb-gradient_tape/dense_17/kernel/Regularizer/MulhuZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_66huZUÖB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb/gradient_tape/dense_17/kernel/Regularizer/Mul_1huZUÖB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb/gradient_tape/conv2d_126/kernel/Regularizer/MulhuZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_10huZUÖB
r
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb1gradient_tape/conv2d_126/kernel/Regularizer/Mul_1huZUÖB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb/gradient_tape/conv2d_129/kernel/Regularizer/MulhuZUÖB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb/gradient_tape/conv2d_132/kernel/Regularizer/MulhuZUÖB
y
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb8gradient_tape/model_5/batch_normalization_59/moments/MulhuZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_26huZUÖB
r
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb1gradient_tape/conv2d_132/kernel/Regularizer/Mul_1huZUÖB
É
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbBgradient_tape/model_5/batch_normalization_58/batchnorm/mul_1/Mul_1huZUÖB
{
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb:gradient_tape/model_5/batch_normalization_58/moments/mul_1huZUÖB
É
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbBgradient_tape/model_5/batch_normalization_59/batchnorm/mul_1/Mul_1huZUÖB
{
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb:gradient_tape/model_5/batch_normalization_59/moments/mul_1huZUÖB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb/gradient_tape/conv2d_135/kernel/Regularizer/MulhuZUÖB
r
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb1gradient_tape/conv2d_135/kernel/Regularizer/Mul_1huZUÖB
Å
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2@8Ä@ÄHÄb@gradient_tape/model_5/batch_normalization_58/batchnorm/mul_1/MulhuZUÖB
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2@8Ä@ÄHÄb.model_5/batch_normalization_59/batchnorm/mul_1huZUÖB
k
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä28Ä@ÄHÄb,gradient_tape/model_5/dropout_34/dropout/MulhuZUÖB
m
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä28Ä@ÄHÄb.gradient_tape/model_5/dropout_34/dropout/Mul_1huZUÖB
m
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä28Ä@ÄHÄb.gradient_tape/model_5/dropout_35/dropout/Mul_1huZUÖB
]
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä28Ä@ÄHÄbmodel_5/dropout_34/dropout/MulhuZUÖB
]
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä28Ä@ÄHÄbmodel_5/dropout_35/dropout/MulhuZUÖB

 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb>gradient_tape/model_5/batch_normalization_58/batchnorm/sub/Neghu  »B

 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb>gradient_tape/model_5/batch_normalization_59/batchnorm/sub/Neghu  »B
q
"Rsqrt_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb.model_5/batch_normalization_58/batchnorm/Rsqrthu  »B
h
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb$conv2d_125/kernel/Regularizer/SquarehuZUÖB
h
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb$conv2d_126/kernel/Regularizer/SquarehuZUÖB
h
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb$conv2d_132/kernel/Regularizer/SquarehuZUÖB
h
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb$conv2d_135/kernel/Regularizer/SquarehuZUÖB
s
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb2model_5/batch_normalization_58/AssignMovingAvg/subhuZUÖB
m
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb,model_5/batch_normalization_58/batchnorm/subhuZUÖB
s
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb2model_5/batch_normalization_59/AssignMovingAvg/subhuZUÖB
m
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb,model_5/batch_normalization_59/batchnorm/subhuZUÖB
y
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2@8Ä@ÄHÄb8gradient_tape/model_5/batch_normalization_58/moments/subhuZUÖB
y
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2@8Ä@ÄHÄb8gradient_tape/model_5/batch_normalization_59/moments/subhuZUÖB
µ
htensorflow::functor::ReluGradHalfKernelVector(Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä28Ä@ÄHÄb,gradient_tape/model_5/activation_64/ReluGradhu  »B
Æ
Ìvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄbmodel_5/dropout_35/dropout/CasthuZUÖB
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_5/conv2d_123/Conv2D/CasthuZUÖB
∞
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_5/conv2d_132/BiasAdd/CasthuZUÖB
∞
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_5/conv2d_133/BiasAdd/CasthuZUÖB
∞
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_5/conv2d_134/BiasAdd/CasthuZUÖB
∞
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_5/conv2d_137/BiasAdd/CasthuZUÖB
∞
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_5/conv2d_138/BiasAdd/CasthuZUÖB
∞
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_5/conv2d_139/BiasAdd/CasthuZUÖB
∞
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_5/conv2d_142/BiasAdd/CasthuZUÖB
Æ
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_5/dense_15/BiasAdd/CasthuZUÖB
«
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄb6gradient_tape/model_5/batch_normalization_58/Cast/CasthuZUÖB
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄbmodel_5/conv2d_129/Conv2D/CasthuZUÖB
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä@ÄHÄbmodel_5/conv2d_127/Conv2D/CasthuZUÖB
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä@ÄHÄbmodel_5/conv2d_138/Conv2D/CasthuZUÖB
£	
Ávoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄbmodel_5/activation_63/ReluhuZUÖB
£	
Ávoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄbmodel_5/activation_64/ReluhuZUÖB
‡
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä28Ä@ÄHÄb;gradient_tape/categorical_crossentropy/weighted_loss/Tile_1huZUÖB
ì
≈void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb,categorical_crossentropy/weighted_loss/valuehuZUÖB
Ò
≈void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb
div_no_nanhuZUÖB
Ω
€void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_rsqrt_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_rsqrt_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb@gradient_tape/model_5/batch_normalization_58/batchnorm/RsqrtGradhuZUÖB
ã
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbCast_6huZUÖB
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb1gradient_tape/model_5/conv2d_123/Conv2D/Cast/CasthuZUÖB
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb1gradient_tape/model_5/conv2d_124/Conv2D/Cast/CasthuZUÖB
∑
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb2gradient_tape/model_5/conv2d_129/BiasAdd/Cast/CasthuZUÖB
∑
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb2gradient_tape/model_5/conv2d_131/BiasAdd/Cast/CasthuZUÖB
∑
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb2gradient_tape/model_5/conv2d_133/BiasAdd/Cast/CasthuZUÖB
∑
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb2gradient_tape/model_5/conv2d_134/BiasAdd/Cast/CasthuZUÖB
∑
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb2gradient_tape/model_5/conv2d_136/BiasAdd/Cast/CasthuZUÖB
∑
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb2gradient_tape/model_5/conv2d_140/BiasAdd/Cast/CasthuZUÖB
∑
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb2gradient_tape/model_5/conv2d_142/BiasAdd/Cast/CasthuZUÖB
µ
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb0gradient_tape/model_5/dense_15/BiasAdd/Cast/CasthuZUÖB
µ
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb0gradient_tape/model_5/dense_16/BiasAdd/Cast/CasthuZUÖB
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb1gradient_tape/model_5/conv2d_125/Conv2D/Cast/CasthuZUÖB
¥
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb/gradient_tape/model_5/dense_17/MatMul/Cast/CasthuZUÖB
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb1gradient_tape/model_5/conv2d_126/Conv2D/Cast/CasthuZUÖB
Ω
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄb8gradient_tape/model_5/batch_normalization_58/Cast_1/CasthuZUÖB
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2P8Ä@ÄHÄb1gradient_tape/model_5/conv2d_135/Conv2D/Cast/CasthuZUÖB
˚
”void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbCast_2huZUÖB
Æ
”void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb9gradient_tape/model_5/batch_normalization_58/moments/CasthuZUÖB
∞
”void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb;gradient_tape/model_5/batch_normalization_58/moments/Cast_1huZUÖB
¡
Òvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb.model_5/batch_normalization_59/AssignMovingAvghuZUÖB
√
Òvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb0model_5/batch_normalization_59/AssignMovingAvg_1huZUÖB
ß
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbAdam/gradients/AddN_11huZUÖB
¶
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbAdam/gradients/AddN_2huZUÖB
ß
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbAdam/gradients/AddN_12huZUÖB
ß
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2P8Ä@ÄHÄbAdam/gradients/AddN_22huZUÖB
ò
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbAssignAddVariableOphuZUÖB
ö
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbAssignAddVariableOp_3huZUÖB
ˆ	
ø	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄbAdam/gradients/AddN_3huZUÖB
Â
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2@8Ä@ÄHÄb@gradient_tape/model_5/batch_normalization_58/moments/BroadcastTohuZUÖB
Á
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2@8Ä@ÄHÄbBgradient_tape/model_5/batch_normalization_58/moments/BroadcastTo_1huZUÖB
Á
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2@8Ä@ÄHÄbBgradient_tape/model_5/batch_normalization_59/moments/BroadcastTo_1huZUÖB
¡	
Ávoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_squared_difference_op<float>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_squared_difference_op<float>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, int) *Ä2@8Ä@ÄHÄb8model_5/batch_normalization_59/moments/SquaredDifferencehuZUÖB
ˇ	
£	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)&*Ä28Ä@ÄHÄb:categorical_crossentropy/softmax_cross_entropy_with_logitshuZUÖB
’
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä28Ä@ÄHÄb0gradient_tape/conv2d_123/kernel/Regularizer/TilehuZUÖB
ê
Ÿvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbAssignAddVariableOp_4huZUÖB
π
Ÿvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb>cond/then/_0/cond/cond/else/_843/cond/cond/AssignAddVariableOphuZUÖB
Ø
Ÿvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb4cond_1/then/_10/cond_1/Adam/Adam/AssignAddVariableOphuZUÖB
≥
ãvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorTupleReducerOp<Eigen::internal::ArgMaxTupleReducer<Eigen::Tuple<long, float> >, Eigen::array<long, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorTupleReducerOp<Eigen::internal::ArgMaxTupleReducer<Eigen::Tuple<long, float> >, Eigen::array<long, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long) *Ä28Ä@ÄHÄbArgMaxhuZUÖB
µ
ãvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorTupleReducerOp<Eigen::internal::ArgMaxTupleReducer<Eigen::Tuple<long, float> >, Eigen::array<long, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorTupleReducerOp<Eigen::internal::ArgMaxTupleReducer<Eigen::Tuple<long, float> >, Eigen::array<long, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long) *Ä28Ä@ÄHÄbArgMax_1huZUÖB
Ö
€void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *Ä28Ä@ÄHÄbAll_12hu¶™¶B
Ö
€void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *Ä28Ä@ÄHÄbAll_18hu¶™¶B
Ö
€void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *Ä28Ä@ÄHÄbAll_20hu¶™¶B
Ö
€void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *Ä28Ä@ÄHÄbAll_28hu¶™¶B
Ö
€void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *Ä28Ä@ÄHÄbAll_34hu¶™¶B
Ö
€void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *Ä28Ä@ÄHÄbAll_36hu¶™¶B
Ö
€void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *Ä28Ä@ÄHÄbAll_42hu¶™¶B
Ö
€void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *Ä28Ä@ÄHÄbAll_44hu¶™¶B
Ö
€void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *Ä28Ä@ÄHÄbAll_48hu¶™¶B
Ö
€void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *Ä28Ä@ÄHÄbAll_50hu¶™¶B
Ö
€void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *Ä28Ä@ÄHÄbAll_52hu¶™¶B
Ö
€void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *Ä28Ä@ÄHÄbAll_56hu¶™¶B
Ö
€void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *Ä28Ä@ÄHÄbAll_60hu¶™¶B
æ
˘void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float>, float>(float*, float*, int, tensorflow::functor::Sum<float>, float)0*Ä28Ä@ÄHÄb!conv2d_127/kernel/Regularizer/Sumhu  »B
æ
˘void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float>, float>(float*, float*, int, tensorflow::functor::Sum<float>, float)0*Ä28Ä@ÄHÄb!conv2d_128/kernel/Regularizer/Sumhu  »B
æ
˘void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float>, float>(float*, float*, int, tensorflow::functor::Sum<float>, float)0*Ä28Ä@ÄHÄb!conv2d_130/kernel/Regularizer/Sumhu  »B
æ
˘void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float>, float>(float*, float*, int, tensorflow::functor::Sum<float>, float)0*Ä28Ä@ÄHÄb!conv2d_134/kernel/Regularizer/Sumhu  »B
æ
˘void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float>, float>(float*, float*, int, tensorflow::functor::Sum<float>, float)0*Ä28Ä@ÄHÄb!conv2d_137/kernel/Regularizer/Sumhu  »B
æ
˘void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float>, float>(float*, float*, int, tensorflow::functor::Sum<float>, float)0*Ä28Ä@ÄHÄb!conv2d_138/kernel/Regularizer/Sumhu  »B
æ
˘void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float>, float>(float*, float*, int, tensorflow::functor::Sum<float>, float)0*Ä28Ä@ÄHÄb!conv2d_139/kernel/Regularizer/Sumhu  »B
æ
˘void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float>, float>(float*, float*, int, tensorflow::functor::Sum<float>, float)0*Ä28Ä@ÄHÄb!conv2d_140/kernel/Regularizer/Sumhu  »B
æ
˘void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float>, float>(float*, float*, int, tensorflow::functor::Sum<float>, float)0*Ä28Ä@ÄHÄb!conv2d_141/kernel/Regularizer/Sumhu  »B
Ö
§void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*) *Ä2Ä8Ä@ÄHÄXb<gradient_tape/model_5/conv2d_132/Conv2D/Conv2DBackpropFilterhu  »B
Ö
§void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*) *Ä2Ä8Ä@ÄHÄXb<gradient_tape/model_5/conv2d_135/Conv2D/Conv2DBackpropFilterhu  »B
ª
évoid tensorflow::(anonymous namespace)::concat_fixed_kernel<bool, int>(tensorflow::GpuDeviceArrayStruct<bool const*, 8>, int, int, int, bool*)*B28Ä@ÄHÄbAll_66/inputhu∞öB
•
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2@8Ä@ÄHÄbmodel_5/dense_15/BiasAddhuZUÖB
•
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2@8Ä@ÄHÄbmodel_5/dense_16/BiasAddhuZUÖB
ê
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb9cond_1/then/_10/cond_1/Adam/Adam/update/ResourceApplyAdamhuZUÖB
í
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb;cond_1/then/_10/cond_1/Adam/Adam/update_1/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb<cond_1/then/_10/cond_1/Adam/Adam/update_14/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb<cond_1/then/_10/cond_1/Adam/Adam/update_27/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb<cond_1/then/_10/cond_1/Adam/Adam/update_38/ResourceApplyAdamhuZUÖB
í
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb;cond_1/then/_10/cond_1/Adam/Adam/update_6/ResourceApplyAdamhuZUÖB
í
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb;cond_1/then/_10/cond_1/Adam/Adam/update_7/ResourceApplyAdamhuZUÖB
í
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb;cond_1/then/_10/cond_1/Adam/Adam/update_9/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb<cond_1/then/_10/cond_1/Adam/Adam/update_46/ResourceApplyAdamhuZUÖB
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_19hu  »B
◊
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_2hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_21hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_22hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_23hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_30hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_31hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_38hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_39hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_45hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_47hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_49hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_51hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_53hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_59hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_63hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_65hu  »B
◊
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_9hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä2 8Ä@ÄHÄbAll_64hu  »B
Î
¬void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*Ä28Ä@ÄHÄbSum_2hu  »B
á
¬void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*Ä28Ä@ÄHÄb!conv2d_125/kernel/Regularizer/Sumhu  »B
á
¬void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*Ä2 8Ä@ÄHÄb!conv2d_126/kernel/Regularizer/Sumhu  »B
á
¬void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*Ä2 8Ä@ÄHÄb!conv2d_132/kernel/Regularizer/Sumhu  »B
á
¬void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*Ä2 8Ä@ÄHÄb!conv2d_135/kernel/Regularizer/Sumhu  »B
â
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2 8Ä@ÄHÄb4gradient_tape/model_5/conv2d_123/BiasAdd/BiasAddGradhuZUÖB
â
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2 8Ä@ÄHÄb4gradient_tape/model_5/conv2d_124/BiasAdd/BiasAddGradhuZUÖB
â
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2@8Ä@ÄHÄb4gradient_tape/model_5/conv2d_125/BiasAdd/BiasAddGradhuZUÖB
ä
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Ä@ÄHÄb4gradient_tape/model_5/conv2d_128/BiasAdd/BiasAddGradhuZUÖB
ÿ
±void tensorflow::functor::CleanupSegments<bool*, bool*, tensorflow::functor::And>(bool*, bool*, int, int, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type)* 28Ä@ÄHÄbAll_32huMUB
◊
±void tensorflow::functor::CleanupSegments<bool*, bool*, tensorflow::functor::And>(bool*, bool*, int, int, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type)* 28Ä@ÄHÄbAll_4huMUB
ÿ
±void tensorflow::functor::CleanupSegments<bool*, bool*, tensorflow::functor::And>(bool*, bool*, int, int, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type)* 28Ä@ÄHÄbAll_64huMUB
á
≈void tensorflow::functor::CleanupSegments<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)* 28Ä@ÄHÄb!conv2d_129/kernel/Regularizer/SumhuMUB
á
≈void tensorflow::functor::CleanupSegments<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)* 28Ä@ÄHÄb!conv2d_132/kernel/Regularizer/SumhuMUB
ö
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2@8Ä@ÄHÄb2gradient_tape/model_5/dense_15/BiasAdd/BiasAddGradhuZUÖB
©
√void tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)Ä!*  2@8Ä@ÄHÄbBgradient_tape/model_5/batch_normalization_58/batchnorm/mul_1/Sum_1huZUÖB
©
√void tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)Ä!*  2@8Ä@ÄHÄbBgradient_tape/model_5/batch_normalization_59/batchnorm/add_1/Sum_1huZUÖB
‘
Åvoid tensorflow::functor::ColumnReduceKernel<float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, tensorflow::functor::Sum<float> >(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)Ä!*  2@8Ä@ÄHÄb/model_5/batch_normalization_58/moments/variancehuZUÖB
–
Åvoid tensorflow::functor::ColumnReduceKernel<float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, tensorflow::functor::Sum<float> >(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)Ä!*  2@8Ä@ÄHÄb+model_5/batch_normalization_59/moments/meanhuZUÖB
‘
Åvoid tensorflow::functor::ColumnReduceKernel<float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, tensorflow::functor::Sum<float> >(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)Ä!*  2@8Ä@ÄHÄb/model_5/batch_normalization_59/moments/variancehuZUÖB
¶
–void tensorflow::functor::ColumnReduceMax16ColumnsKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿* 28Ä@ÄHÄb2gradient_tape/model_5/dense_17/BiasAdd/BiasAddGradhu  ØB
ñ
◊void tensorflow::functor::RowReduceKernel<cub::TransformInputIterator<float, tensorflow::(anonymous namespace)::SubtractAndExpFunctor<Eigen::half, float>, cub::CountingInputIterator<int, long>, long>, float*, cub::Sum>(cub::TransformInputIterator<float, tensorflow::(anonymous namespace)::SubtractAndExpFunctor<Eigen::half, float>, cub::CountingInputIterator<int, long>, long>, float*, int, int, cub::Sum, std::iterator_traits<cub::TransformInputIterator<float, tensorflow::(anonymous namespace)::SubtractAndExpFunctor<Eigen::half, float>, cub::CountingInputIterator<int, long>, long> >::value_type)*Ä28Ä@ÄHÄbmodel_5/activation_65/Softmaxhu  »B
‘
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä28Ä@ÄHÄXbmodel_5/conv2d_125/Conv2DhuZUÖB
˜
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä28Ä@ÄHÄXb<gradient_tape/model_5/conv2d_125/Conv2D/Conv2DBackpropFilterhuZUÖB
`
 Pow_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Å@ÅHÅbcond_1/then/_10/cond_1/Adam/PowhuZUÖB
o
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb,model_5/batch_normalization_59/batchnorm/addhuZUÖB
}
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2@8Ä@ÄHÄb<gradient_tape/model_5/batch_normalization_58/moments/truedivhuZUÖB

 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2@8Ä@ÄHÄb>gradient_tape/model_5/batch_normalization_58/moments/truediv_1huZUÖB
}
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2@8Ä@ÄHÄb<gradient_tape/model_5/batch_normalization_59/moments/truedivhuZUÖB
G
!Equal_GPU_DT_INT64_DT_BOOL_kernel*Ä28Ä@ÄHÄbEqualhuZUÖB
o
'GreaterEqual_GPU_DT_HALF_DT_BOOL_kernel*Ä28Ä@ÄHÄb'model_5/dropout_35/dropout/GreaterEqualhuZUÖB
M
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinitehu  »B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄb
IsFinite_1hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_13hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_14hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_17hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_21hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_23hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_25hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_27hu  »B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄb
IsFinite_3hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_30hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_33hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_38hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_41hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_45hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_49hu  »B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄb
IsFinite_5hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_51hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_65hu  »B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄb
IsFinite_7hu  »B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄb
IsFinite_9hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_47hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_57hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_58hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_61hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_62hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_55hu  »B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄb
IsFinite_4hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_64hu  »B
P
%LogicalAnd_GPU_DT_BOOL_DT_BOOL_kernel*Ä28Ä@ÄHÄb
LogicalAndhuZUÖB
ç
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbLgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mulhuZUÖB
D
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbMulhuZUÖB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb!conv2d_123/kernel/Regularizer/mulhuZUÖB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb!conv2d_124/kernel/Regularizer/mulhuZUÖB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb!conv2d_125/kernel/Regularizer/mulhuZUÖB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb!conv2d_126/kernel/Regularizer/mulhuZUÖB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb!conv2d_129/kernel/Regularizer/mulhuZUÖB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb!conv2d_130/kernel/Regularizer/mulhuZUÖB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb!conv2d_131/kernel/Regularizer/mulhuZUÖB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb!conv2d_132/kernel/Regularizer/mulhuZUÖB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb!conv2d_135/kernel/Regularizer/mulhuZUÖB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb!conv2d_137/kernel/Regularizer/mulhuZUÖB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb!conv2d_138/kernel/Regularizer/mulhuZUÖB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb!conv2d_140/kernel/Regularizer/mulhuZUÖB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb!conv2d_141/kernel/Regularizer/mulhuZUÖB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb!conv2d_142/kernel/Regularizer/mulhuZUÖB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb!conv2d_143/kernel/Regularizer/mulhuZUÖB
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbdense_15/kernel/Regularizer/mulhuZUÖB
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbdense_16/kernel/Regularizer/mulhuZUÖB
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbdense_17/kernel/Regularizer/mulhuZUÖB
v
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb5gradient_tape/conv2d_123/kernel/Regularizer/mul/Mul_1huZUÖB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb/gradient_tape/conv2d_124/kernel/Regularizer/MulhuZUÖB
Å
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb@gradient_tape/model_5/batch_normalization_59/batchnorm/mul/Mul_1huZUÖB
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb2model_5/batch_normalization_58/AssignMovingAvg/mulhuZUÖB
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb4model_5/batch_normalization_58/AssignMovingAvg_1/mulhuZUÖB
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb2model_5/batch_normalization_59/AssignMovingAvg/mulhuZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_11huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_13huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_16huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_17huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_19huZUÖB
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_2huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_21huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_23huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_24huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_25huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_29huZUÖB
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_3huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_31huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_33huZUÖB
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_4huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_40huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_41huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_43huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_45huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_47huZUÖB
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_5huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_55huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_59huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_61huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_67huZUÖB
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_7huZUÖB
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_8huZUÖB
r
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb1gradient_tape/conv2d_123/kernel/Regularizer/Mul_1huZUÖB

 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb>gradient_tape/model_5/batch_normalization_58/batchnorm/mul/MulhuZUÖB
É
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbBgradient_tape/model_5/batch_normalization_58/batchnorm/mul_2/Mul_1huZUÖB

 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb>gradient_tape/model_5/batch_normalization_59/batchnorm/mul/MulhuZUÖB
m
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb,model_5/batch_normalization_59/batchnorm/mulhuZUÖB
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb.model_5/batch_normalization_59/batchnorm/mul_2huZUÖB
y
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb8gradient_tape/model_5/batch_normalization_58/moments/MulhuZUÖB
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2@8Ä@ÄHÄb.model_5/batch_normalization_58/batchnorm/mul_1huZUÖB
k
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä28Ä@ÄHÄb,gradient_tape/model_5/dropout_35/dropout/MulhuZUÖB
_
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä28Ä@ÄHÄb model_5/dropout_34/dropout/Mul_1huZUÖB
_
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä28Ä@ÄHÄb model_5/dropout_35/dropout/Mul_1huZUÖB
b
 Pow_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb!cond_1/then/_10/cond_1/Adam/Pow_1huZUÖB
q
"Rsqrt_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb.model_5/batch_normalization_59/batchnorm/Rsqrthu  »B
h
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb$conv2d_123/kernel/Regularizer/SquarehuZUÖB
h
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb$conv2d_124/kernel/Regularizer/SquarehuZUÖB
f
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb"dense_17/kernel/Regularizer/SquarehuZUÖB
u
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb4model_5/batch_normalization_58/AssignMovingAvg_1/subhuZUÖB
u
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb4model_5/batch_normalization_59/AssignMovingAvg_1/subhuZUÖB
Æ
Ìvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄbmodel_5/dropout_34/dropout/CasthuZUÖB
é
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb}categorical_crossentropy/softmax_cross_entropy_with_logits/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Cast_2huZUÖB
æ
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb-categorical_crossentropy/weighted_loss/Cast_1huZUÖB
Ê
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbUgradient_tape/Cast_3/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_CasthuZUÖB
†
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbégradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/Cast/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_CasthuZUÖB
ã
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbzgradient_tape/categorical_crossentropy/weighted_loss/Cast/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_CasthuZUÖB
∞
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_5/conv2d_123/BiasAdd/CasthuZUÖB
∞
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_5/conv2d_124/BiasAdd/CasthuZUÖB
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_5/conv2d_124/Conv2D/CasthuZUÖB
∞
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_5/conv2d_125/BiasAdd/CasthuZUÖB
∞
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_5/conv2d_126/BiasAdd/CasthuZUÖB
∞
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_5/conv2d_127/BiasAdd/CasthuZUÖB
∞
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_5/conv2d_128/BiasAdd/CasthuZUÖB
∞
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_5/conv2d_129/BiasAdd/CasthuZUÖB
∞
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_5/conv2d_130/BiasAdd/CasthuZUÖB
∞
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_5/conv2d_131/BiasAdd/CasthuZUÖB
∞
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_5/conv2d_135/BiasAdd/CasthuZUÖB
∞
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_5/conv2d_136/BiasAdd/CasthuZUÖB
∞
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_5/conv2d_140/BiasAdd/CasthuZUÖB
∞
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_5/conv2d_141/BiasAdd/CasthuZUÖB
∞
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_5/conv2d_143/BiasAdd/CasthuZUÖB
Æ
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_5/dense_17/BiasAdd/CasthuZUÖB
Æ
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_5/dense_16/BiasAdd/CasthuZUÖB
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_5/conv2d_125/Conv2D/CasthuZUÖB
≠
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_5/dense_17/MatMul/CasthuZUÖB
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_5/conv2d_126/Conv2D/CasthuZUÖB
«
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄb6gradient_tape/model_5/batch_normalization_59/Cast/CasthuZUÖB
∂
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄb%model_5/batch_normalization_58/Cast_1huZUÖB
∂
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄb%model_5/batch_normalization_59/Cast_1huZUÖB
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄbmodel_5/conv2d_132/Conv2D/CasthuZUÖB
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2P8Ä@ÄHÄbmodel_5/conv2d_135/Conv2D/CasthuZUÖB
Û
≈void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbdiv_no_nan_1huZUÖB
¨
≈void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbEgradient_tape/categorical_crossentropy/weighted_loss/value/div_no_nanhuZUÖB
Ω
€void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_rsqrt_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_rsqrt_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb@gradient_tape/model_5/batch_normalization_59/batchnorm/RsqrtGradhuZUÖB
¬
ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_inverse_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_inverse_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä28Ä@ÄHÄbtruedivhuZUÖB
ã
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbCast_1huZUÖB
Ç
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb}categorical_crossentropy/softmax_cross_entropy_with_logits/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Cast_1huZUÖB
ƒ
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb?categorical_crossentropy/softmax_cross_entropy_with_logits/CasthuZUÖB
∆
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbAcategorical_crossentropy/softmax_cross_entropy_with_logits/Cast_1huZUÖB
Ï
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbgcategorical_crossentropy/weighted_loss/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_CasthuZUÖB
ñ
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbêgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/Cast_2/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_CasthuZUÖB
≈
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb@gradient_tape/categorical_crossentropy/weighted_loss/Cast_1/CasthuZUÖB
∑
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb2gradient_tape/model_5/conv2d_123/BiasAdd/Cast/CasthuZUÖB
∑
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb2gradient_tape/model_5/conv2d_124/BiasAdd/Cast/CasthuZUÖB
∑
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb2gradient_tape/model_5/conv2d_125/BiasAdd/Cast/CasthuZUÖB
∑
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb2gradient_tape/model_5/conv2d_126/BiasAdd/Cast/CasthuZUÖB
∑
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb2gradient_tape/model_5/conv2d_127/BiasAdd/Cast/CasthuZUÖB
∑
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb2gradient_tape/model_5/conv2d_128/BiasAdd/Cast/CasthuZUÖB
∑
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb2gradient_tape/model_5/conv2d_130/BiasAdd/Cast/CasthuZUÖB
∑
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb2gradient_tape/model_5/conv2d_135/BiasAdd/Cast/CasthuZUÖB
∑
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb2gradient_tape/model_5/conv2d_137/BiasAdd/Cast/CasthuZUÖB
∑
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb2gradient_tape/model_5/conv2d_138/BiasAdd/Cast/CasthuZUÖB
∑
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb2gradient_tape/model_5/conv2d_139/BiasAdd/Cast/CasthuZUÖB
∑
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb2gradient_tape/model_5/conv2d_141/BiasAdd/Cast/CasthuZUÖB
∑
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb2gradient_tape/model_5/conv2d_143/BiasAdd/Cast/CasthuZUÖB
µ
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb0gradient_tape/model_5/dense_17/BiasAdd/Cast/CasthuZUÖB
Ω
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄb8gradient_tape/model_5/batch_normalization_59/Cast_1/CasthuZUÖB
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄb1gradient_tape/model_5/conv2d_132/Conv2D/Cast/CasthuZUÖB
®
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄb#model_5/batch_normalization_58/CasthuZUÖB
®
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄb#model_5/batch_normalization_59/CasthuZUÖB
˝
’void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbCast_7huZUÖB
˚
”void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbCast_8huZUÖB
≠
”void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb8categorical_crossentropy/weighted_loss/num_elements/CasthuZUÖB
Æ
”void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb9gradient_tape/model_5/batch_normalization_59/moments/CasthuZUÖB
∞
”void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb;gradient_tape/model_5/batch_normalization_59/moments/Cast_1huZUÖB
ô
’void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb"cond_1/then/_10/cond_1/Adam/Cast_1huZUÖB
¡
Òvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb.model_5/batch_normalization_58/AssignMovingAvghuZUÖB
√
Òvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb0model_5/batch_normalization_58/AssignMovingAvg_1huZUÖB
ß
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbAdam/gradients/AddN_10huZUÖB
§
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbAdam/gradients/AddNhuZUÖB
ß
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbAdam/gradients/AddN_33huZUÖB
ö
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbAssignAddVariableOp_1huZUÖB
ö
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbAssignAddVariableOp_2huZUÖB
È
üvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long) *Ä28Ä@ÄHÄb(ArithmeticOptimizer/AddOpsRewrite_AddN_1huZUÖB
˜
õvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorEvalToOp<Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_log_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorEvalToOp<Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_log_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb:categorical_crossentropy/softmax_cross_entropy_with_logitshuZUÖB
Ö
€void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *Ä28Ä@ÄHÄbAll_10hu¶™¶B
Ö
€void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *Ä28Ä@ÄHÄbAll_26hu¶™¶B