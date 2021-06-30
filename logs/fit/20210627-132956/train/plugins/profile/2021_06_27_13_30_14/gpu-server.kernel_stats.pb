
ô
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8Ìß@ÌßHÌßb<cond_1/then/_10/cond_1/Adam/Adam/update_56/ResourceApplyAdamhuZUÖB
¯
ï_ZN13cutlass_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi64ELi256ELi32EEENS4_51Conv2dWgradOutputGradientTileAccessIteratorAnalyticINS_11MatrixShapeILi64ELi32EEENS_6half_tENS_9transform29PitchLinearWarpRakedThreadMapINS_6layout16PitchLinearShapeILi64ELi32EEELi128ENSG_ILi8ELi4EEELi8EEEEENSD_11threadblock25RegularTileAccessIteratorISB_SC_NSF_40ColumnMajorTensorOpMultiplicandCongruousILi16ELi64EEELi1ESJ_Li16EEELNS_4arch14CacheOperation4KindE0ENS4_47Conv2dWgradActivationTileAccessIteratorAnalyticINSA_ILi32ELi256EEESC_NSE_INSG_ILi256ELi32EEELi128ESI_Li8EEEEENSM_ISU_SC_NSF_37RowMajorTensorOpMultiplicandCongruousILi16ELi64EEELi0ESW_Li16EEELSS_0ENS6_11threadblock9MmaPolicyINS6_4warp11MmaTensorOpINS7_ILi64ELi64ELi32EEESC_SO_SC_SZ_fNSF_8RowMajorENS13_17MmaTensorOpPolicyINSQ_3MmaINS7_ILi16ELi8ELi16EEELi32ESC_S16_SC_NSF_11ColumnMajorEfS16_NSQ_13OpMultiplyAddEEENSA_ILi1ELi1EEEEELi1ELb0EbEENSA_ILi0ELi0EEES1G_Li1EEELi4EbEENS_8epilogue11threadblock8EpilogueIS8_S1F_Li1ENS1K_22PredicatedTileIteratorINS1K_26OutputTileOptimalThreadMapINS1K_15OutputTileShapeILi256ELi8ELi1ELi1ELi1EEENS1O_ILi1ELi8ELi1ELi1ELi8EEELi128ELi4ELi32EEEfEENS1J_4warp24FragmentIteratorTensorOpIS15_S19_fNS_5ArrayIfLi4ELb1EEES16_EENS1T_20TileIteratorTensorOpIS15_S19_fS16_EENS1K_18SharedLoadIteratorINS1R_18CompactedThreadMapEfLi16EEENS1J_6thread17LinearCombinationIfLi4EffLNS_15FloatRoundStyleE2EEENSA_ILi0ELi8EEEEENS11_30GemmIdentityThreadblockSwizzleILi4EEELNS1_8OperatorE2ENS1_17Conv2dProblemSizeEEEEEvNT_6ParamsEˇ ÄÄ*Ä22)8ø∏é@ø∏éHø∏éXb9gradient_tape/model/conv2d_11/Conv2D/Conv2DBackpropFilterh
æ
|sm80_xmma_fprop_implicit_gemm_f16f16_f16f32_f32_nhwckrsc_nhwc_tilesize256x64x32_stage3_warpsize4x1x1_g1_tensor16x8x16_kernel˙ Ä‡*Ä2Ä8∑–◊@∑–◊H∑–◊PXbmodel/conv2d_11/Conv2Dh
ô
∂void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride>(cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride::Params)ı Ä¿*Ä2Ä8∑à÷@∑à÷H∑à÷Xb8gradient_tape/model/conv2d_11/Conv2D/Conv2DBackpropInputh
ô
∂void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride>(cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride::Params)ı Ä¿*Ä2Ä8´êé@´êéH´êéXb8gradient_tape/model/conv2d_17/Conv2D/Conv2DBackpropInputh
≠
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8™¯â@™¯âH™¯âbAdam/gradients/AddN_31huZUÖB
›
úvoid cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3>(cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3::Params)˛ Ä¿*Ä2Ä8™®à@™®àH™®àXbmodel/conv2d_17/Conv2Dh
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ä¿8™†Ñ@™†ÑH™†Ñb,gradient_tape/dense/kernel/Regularizer/Mul_1huZUÖB
√
ﬁvoid xmma_cudnn::gemm::kernel<xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, xmma_cudnn::Col, 256, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 3> >(xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, xmma_cudnn::Col, 256, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 3>::Params)Ó Ä¿*Ä2$	8©∏Å@©∏ÅH©∏ÅPXb9gradient_tape/model/conv2d_10/Conv2D/Conv2DBackpropFilterh
¯
ï_ZN13cutlass_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi64ELi256ELi32EEENS4_51Conv2dWgradOutputGradientTileAccessIteratorAnalyticINS_11MatrixShapeILi64ELi32EEENS_6half_tENS_9transform29PitchLinearWarpRakedThreadMapINS_6layout16PitchLinearShapeILi64ELi32EEELi128ENSG_ILi8ELi4EEELi8EEEEENSD_11threadblock25RegularTileAccessIteratorISB_SC_NSF_40ColumnMajorTensorOpMultiplicandCongruousILi16ELi64EEELi1ESJ_Li16EEELNS_4arch14CacheOperation4KindE0ENS4_47Conv2dWgradActivationTileAccessIteratorAnalyticINSA_ILi32ELi256EEESC_NSE_INSG_ILi256ELi32EEELi128ESI_Li8EEEEENSM_ISU_SC_NSF_37RowMajorTensorOpMultiplicandCongruousILi16ELi64EEELi0ESW_Li16EEELSS_0ENS6_11threadblock9MmaPolicyINS6_4warp11MmaTensorOpINS7_ILi64ELi64ELi32EEESC_SO_SC_SZ_fNSF_8RowMajorENS13_17MmaTensorOpPolicyINSQ_3MmaINS7_ILi16ELi8ELi16EEELi32ESC_S16_SC_NSF_11ColumnMajorEfS16_NSQ_13OpMultiplyAddEEENSA_ILi1ELi1EEEEELi1ELb0EbEENSA_ILi0ELi0EEES1G_Li1EEELi4EbEENS_8epilogue11threadblock8EpilogueIS8_S1F_Li1ENS1K_22PredicatedTileIteratorINS1K_26OutputTileOptimalThreadMapINS1K_15OutputTileShapeILi256ELi8ELi1ELi1ELi1EEENS1O_ILi1ELi8ELi1ELi1ELi8EEELi128ELi4ELi32EEEfEENS1J_4warp24FragmentIteratorTensorOpIS15_S19_fNS_5ArrayIfLi4ELb1EEES16_EENS1T_20TileIteratorTensorOpIS15_S19_fS16_EENS1K_18SharedLoadIteratorINS1R_18CompactedThreadMapEfLi16EEENS1J_6thread17LinearCombinationIfLi4EffLNS_15FloatRoundStyleE2EEENSA_ILi0ELi8EEEEENS11_30GemmIdentityThreadblockSwizzleILi4EEELNS1_8OperatorE2ENS1_17Conv2dProblemSizeEEEEEvNT_6ParamsEˇ ÄÄ*Ä2
8®ò˝@®ò˝H®ò˝Xb9gradient_tape/model/conv2d_17/Conv2D/Conv2DBackpropFilterh
›
úvoid cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3>(cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3::Params)˛ Ä¿*Ä2Ä8®‡˙@®‡˙H®‡˙Xbmodel/conv2d_10/Conv2Dh
ô
∂void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride>(cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride::Params)ı Ä¿*Ä2Ä8®∞¯@®∞¯H®∞¯Xb8gradient_tape/model/conv2d_10/Conv2D/Conv2DBackpropInputh
›
úvoid cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3>(cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3::Params)˛ Ä¿*Ä2Ä8°‡“@°‡“H°‡“Xbmodel/conv2d_14/Conv2Dh
¯
ï_ZN13cutlass_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi64ELi256ELi32EEENS4_51Conv2dWgradOutputGradientTileAccessIteratorAnalyticINS_11MatrixShapeILi64ELi32EEENS_6half_tENS_9transform29PitchLinearWarpRakedThreadMapINS_6layout16PitchLinearShapeILi64ELi32EEELi128ENSG_ILi8ELi4EEELi8EEEEENSD_11threadblock25RegularTileAccessIteratorISB_SC_NSF_40ColumnMajorTensorOpMultiplicandCongruousILi16ELi64EEELi1ESJ_Li16EEELNS_4arch14CacheOperation4KindE0ENS4_47Conv2dWgradActivationTileAccessIteratorAnalyticINSA_ILi32ELi256EEESC_NSE_INSG_ILi256ELi32EEELi128ESI_Li8EEEEENSM_ISU_SC_NSF_37RowMajorTensorOpMultiplicandCongruousILi16ELi64EEELi0ESW_Li16EEELSS_0ENS6_11threadblock9MmaPolicyINS6_4warp11MmaTensorOpINS7_ILi64ELi64ELi32EEESC_SO_SC_SZ_fNSF_8RowMajorENS13_17MmaTensorOpPolicyINSQ_3MmaINS7_ILi16ELi8ELi16EEELi32ESC_S16_SC_NSF_11ColumnMajorEfS16_NSQ_13OpMultiplyAddEEENSA_ILi1ELi1EEEEELi1ELb0EbEENSA_ILi0ELi0EEES1G_Li1EEELi4EbEENS_8epilogue11threadblock8EpilogueIS8_S1F_Li1ENS1K_22PredicatedTileIteratorINS1K_26OutputTileOptimalThreadMapINS1K_15OutputTileShapeILi256ELi8ELi1ELi1ELi1EEENS1O_ILi1ELi8ELi1ELi1ELi8EEELi128ELi4ELi32EEEfEENS1J_4warp24FragmentIteratorTensorOpIS15_S19_fNS_5ArrayIfLi4ELb1EEES16_EENS1T_20TileIteratorTensorOpIS15_S19_fS16_EENS1K_18SharedLoadIteratorINS1R_18CompactedThreadMapEfLi16EEENS1J_6thread17LinearCombinationIfLi4EffLNS_15FloatRoundStyleE2EEENSA_ILi0ELi8EEEEENS11_30GemmIdentityThreadblockSwizzleILi4EEELNS1_8OperatorE2ENS1_17Conv2dProblemSizeEEEEEvNT_6ParamsEˇ ÄÄ*Ä2
8†ÄÃ@†ÄÃH†ÄÃXb9gradient_tape/model/conv2d_14/Conv2D/Conv2DBackpropFilterh
›
úvoid cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3>(cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3::Params)˛ Ä¿*Ä2Ä8ü®¬@ü®¬Hü®¬Xbmodel/conv2d_22/Conv2Dh
ô
∂void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride>(cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride::Params)ı Ä¿*Ä2Ä8û¯º@û¯ºHû¯ºXb8gradient_tape/model/conv2d_14/Conv2D/Conv2DBackpropInputh
√
ﬁvoid xmma_cudnn::gemm::kernel<xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, xmma_cudnn::Col, 256, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 3> >(xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, xmma_cudnn::Col, 256, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 3>::Params)Ó Ä¿*Ä2Z8ùà¥@ùà¥Hùà¥PXb9gradient_tape/model/conv2d_22/Conv2D/Conv2DBackpropFilterh
±
Œvoid cudnn::ops::pooling_bw_kernel_max<__half, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, false>(cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor) Ä *Ä2Ä 8ù¿≤@ù¿≤Hù¿≤b7gradient_tape/model/max_pooling2d_1/MaxPool/MaxPoolGradhu  »B
O
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ä¿8õ‡Æ@õ‡ÆHõ‡Æbmul_58huZUÖB
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ä¿8ú–Æ@ú–ÆHú–Æb*gradient_tape/dense/kernel/Regularizer/MulhuZUÖB
k
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ä¿8ú†Æ@ú†ÆHú†Æbdense/kernel/Regularizer/SquarehuZUÖB
µ
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8õ≠@õ≠Hõ≠b*gradient_tape/model/dense/MatMul/Cast/CasthuZUÖB
Æ
Œvoid cudnn::ops::pooling_bw_kernel_max<__half, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, false>(cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor) Ä *Ä2   8õ∞ß@õ∞ßHõ∞ßb5gradient_tape/model/max_pooling2d/MaxPool/MaxPoolGradhu  »B
ô
∂void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride>(cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride::Params)ı Ä¿*Ä2Ä8ô®õ@ô®õHô®õXb8gradient_tape/model/conv2d_22/Conv2D/Conv2DBackpropInputh
Æ
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8ô–ö@ô–öHô–öbmodel/dense/MatMul/CasthuZUÖB
Ω
|sm80_xmma_fprop_implicit_gemm_f16f16_f16f32_f32_nhwckrsc_nhwc_tilesize256x64x32_stage3_warpsize4x1x1_g1_tensor16x8x16_kernel˙ Ä‡*Ä2 8ñÜ@ñÜHñÜPXbmodel/conv2d_23/Conv2Dh
¯
ï_ZN13cutlass_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi64ELi256ELi32EEENS4_51Conv2dWgradOutputGradientTileAccessIteratorAnalyticINS_11MatrixShapeILi64ELi32EEENS_6half_tENS_9transform29PitchLinearWarpRakedThreadMapINS_6layout16PitchLinearShapeILi64ELi32EEELi128ENSG_ILi8ELi4EEELi8EEEEENSD_11threadblock25RegularTileAccessIteratorISB_SC_NSF_40ColumnMajorTensorOpMultiplicandCongruousILi16ELi64EEELi1ESJ_Li16EEELNS_4arch14CacheOperation4KindE0ENS4_47Conv2dWgradActivationTileAccessIteratorAnalyticINSA_ILi32ELi256EEESC_NSE_INSG_ILi256ELi32EEELi128ESI_Li8EEEEENSM_ISU_SC_NSF_37RowMajorTensorOpMultiplicandCongruousILi16ELi64EEELi0ESW_Li16EEELSS_0ENS6_11threadblock9MmaPolicyINS6_4warp11MmaTensorOpINS7_ILi64ELi64ELi32EEESC_SO_SC_SZ_fNSF_8RowMajorENS13_17MmaTensorOpPolicyINSQ_3MmaINS7_ILi16ELi8ELi16EEELi32ESC_S16_SC_NSF_11ColumnMajorEfS16_NSQ_13OpMultiplyAddEEENSA_ILi1ELi1EEEEELi1ELb0EbEENSA_ILi0ELi0EEES1G_Li1EEELi4EbEENS_8epilogue11threadblock8EpilogueIS8_S1F_Li1ENS1K_22PredicatedTileIteratorINS1K_26OutputTileOptimalThreadMapINS1K_15OutputTileShapeILi256ELi8ELi1ELi1ELi1EEENS1O_ILi1ELi8ELi1ELi1ELi8EEELi128ELi4ELi32EEEfEENS1J_4warp24FragmentIteratorTensorOpIS15_S19_fNS_5ArrayIfLi4ELb1EEES16_EENS1T_20TileIteratorTensorOpIS15_S19_fS16_EENS1K_18SharedLoadIteratorINS1R_18CompactedThreadMapEfLi16EEENS1J_6thread17LinearCombinationIfLi4EffLNS_15FloatRoundStyleE2EEENSA_ILi0ELi8EEEEENS11_30GemmIdentityThreadblockSwizzleILi4EEELNS1_8OperatorE2ENS1_17Conv2dProblemSizeEEEEEvNT_6ParamsEˇ ÄÄ*Ä2 8ïÿÖ@ïÿÖHïÿÖXb9gradient_tape/model/conv2d_23/Conv2D/Conv2DBackpropFilterh
ñ
∂void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride>(cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride::Params)ı Ä¿*Ä2Ä8ë†l@ë†lHë†lXb8gradient_tape/model/conv2d_23/Conv2D/Conv2DBackpropInputh
U
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2ÄÄ8ë»i@ë»iHë»ibIsFinite_56hu  »B
”
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8ë¿i@ë¿iHë¿ib+gradient_tape/dense/kernel/Regularizer/TilehuZUÖB
ø
ﬁvoid xmma_cudnn::gemm::kernel<xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 32, 2, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 32, 2, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 32, 2, 2, 1, 1>, 16, xmma_cudnn::Col, 128, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 32, 2, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 32, 2, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 4> >(xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 32, 2, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 32, 2, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 32, 2, 2, 1, 1>, 16, xmma_cudnn::Col, 128, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 32, 2, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 32, 2, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 4>::Params)Ù ÄÄ*Ä2	8ë¯f@ë¯fHë¯fPXb8gradient_tape/model/conv2d_8/Conv2D/Conv2DBackpropFilterh
ñ
∂void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride>(cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride::Params)ı Ä¿*Ä2Ä8èÿc@èÿcHèÿcXb8gradient_tape/model/conv2d_16/Conv2D/Conv2DBackpropInputh
⁄
úvoid cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3>(cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3::Params)˛ Ä¿*Ä2Ä8è–b@è–bHè–bXbmodel/conv2d_16/Conv2Dh
ı
ï_ZN13cutlass_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi64ELi256ELi32EEENS4_51Conv2dWgradOutputGradientTileAccessIteratorAnalyticINS_11MatrixShapeILi64ELi32EEENS_6half_tENS_9transform29PitchLinearWarpRakedThreadMapINS_6layout16PitchLinearShapeILi64ELi32EEELi128ENSG_ILi8ELi4EEELi8EEEEENSD_11threadblock25RegularTileAccessIteratorISB_SC_NSF_40ColumnMajorTensorOpMultiplicandCongruousILi16ELi64EEELi1ESJ_Li16EEELNS_4arch14CacheOperation4KindE0ENS4_47Conv2dWgradActivationTileAccessIteratorAnalyticINSA_ILi32ELi256EEESC_NSE_INSG_ILi256ELi32EEELi128ESI_Li8EEEEENSM_ISU_SC_NSF_37RowMajorTensorOpMultiplicandCongruousILi16ELi64EEELi0ESW_Li16EEELSS_0ENS6_11threadblock9MmaPolicyINS6_4warp11MmaTensorOpINS7_ILi64ELi64ELi32EEESC_SO_SC_SZ_fNSF_8RowMajorENS13_17MmaTensorOpPolicyINSQ_3MmaINS7_ILi16ELi8ELi16EEELi32ESC_S16_SC_NSF_11ColumnMajorEfS16_NSQ_13OpMultiplyAddEEENSA_ILi1ELi1EEEEELi1ELb0EbEENSA_ILi0ELi0EEES1G_Li1EEELi4EbEENS_8epilogue11threadblock8EpilogueIS8_S1F_Li1ENS1K_22PredicatedTileIteratorINS1K_26OutputTileOptimalThreadMapINS1K_15OutputTileShapeILi256ELi8ELi1ELi1ELi1EEENS1O_ILi1ELi8ELi1ELi1ELi8EEELi128ELi4ELi32EEEfEENS1J_4warp24FragmentIteratorTensorOpIS15_S19_fNS_5ArrayIfLi4ELb1EEES16_EENS1T_20TileIteratorTensorOpIS15_S19_fS16_EENS1K_18SharedLoadIteratorINS1R_18CompactedThreadMapEfLi16EEENS1J_6thread17LinearCombinationIfLi4EffLNS_15FloatRoundStyleE2EEENSA_ILi0ELi8EEEEENS11_30GemmIdentityThreadblockSwizzleILi4EEELNS1_8OperatorE2ENS1_17Conv2dProblemSizeEEEEEvNT_6ParamsEˇ ÄÄ*Ä2
8êàb@êàbHêàbXb9gradient_tape/model/conv2d_16/Conv2D/Conv2DBackpropFilterh
Ÿ
úvoid cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3>(cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3::Params)˛ Ä¿*Ä2Ä8èË^@èË^HèË^Xbmodel/conv2d_8/Conv2Dh
Ø
Œvoid xmma_cudnn::gemm::kernel<xmma_cudnn::implicit_gemm::dgrad::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_a_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 16, false, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 16, xmma_cudnn::Row, 32, 256> >, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_c_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, 16, xmma_cudnn::Fragment_c<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, false>, false>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 3> >(xmma_cudnn::implicit_gemm::dgrad::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_a_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 16, false, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 16, xmma_cudnn::Row, 32, 256> >, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_c_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, 16, xmma_cudnn::Fragment_c<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, false>, false>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 3>::Params)ˇ Ä‡*Ä2Ä8èò]@èò]Hèò]PXb7gradient_tape/model/conv2d_8/Conv2D/Conv2DBackpropInputh
ø
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2ú8çR@çRHçRbdense/kernel/Regularizer/Sumhu  »B
ï
∂void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride>(cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride::Params)ı Ä¿*Ä2@8çòQ@çòQHçòQXb8gradient_tape/model/conv2d_20/Conv2D/Conv2DBackpropInputh
⁄
úvoid cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3>(cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3::Params)˛ Ä¿*Ä2Ä8å∏M@å∏MHå∏MXbmodel/conv2d_13/Conv2Dh
ı
ï_ZN13cutlass_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi64ELi256ELi32EEENS4_51Conv2dWgradOutputGradientTileAccessIteratorAnalyticINS_11MatrixShapeILi64ELi32EEENS_6half_tENS_9transform29PitchLinearWarpRakedThreadMapINS_6layout16PitchLinearShapeILi64ELi32EEELi128ENSG_ILi8ELi4EEELi8EEEEENSD_11threadblock25RegularTileAccessIteratorISB_SC_NSF_40ColumnMajorTensorOpMultiplicandCongruousILi16ELi64EEELi1ESJ_Li16EEELNS_4arch14CacheOperation4KindE0ENS4_47Conv2dWgradActivationTileAccessIteratorAnalyticINSA_ILi32ELi256EEESC_NSE_INSG_ILi256ELi32EEELi128ESI_Li8EEEEENSM_ISU_SC_NSF_37RowMajorTensorOpMultiplicandCongruousILi16ELi64EEELi0ESW_Li16EEELSS_0ENS6_11threadblock9MmaPolicyINS6_4warp11MmaTensorOpINS7_ILi64ELi64ELi32EEESC_SO_SC_SZ_fNSF_8RowMajorENS13_17MmaTensorOpPolicyINSQ_3MmaINS7_ILi16ELi8ELi16EEELi32ESC_S16_SC_NSF_11ColumnMajorEfS16_NSQ_13OpMultiplyAddEEENSA_ILi1ELi1EEEEELi1ELb0EbEENSA_ILi0ELi0EEES1G_Li1EEELi4EbEENS_8epilogue11threadblock8EpilogueIS8_S1F_Li1ENS1K_22PredicatedTileIteratorINS1K_26OutputTileOptimalThreadMapINS1K_15OutputTileShapeILi256ELi8ELi1ELi1ELi1EEENS1O_ILi1ELi8ELi1ELi1ELi8EEELi128ELi4ELi32EEEfEENS1J_4warp24FragmentIteratorTensorOpIS15_S19_fNS_5ArrayIfLi4ELb1EEES16_EENS1T_20TileIteratorTensorOpIS15_S19_fS16_EENS1K_18SharedLoadIteratorINS1R_18CompactedThreadMapEfLi16EEENS1J_6thread17LinearCombinationIfLi4EffLNS_15FloatRoundStyleE2EEENSA_ILi0ELi8EEEEENS11_30GemmIdentityThreadblockSwizzleILi4EEELNS1_8OperatorE2ENS1_17Conv2dProblemSizeEEEEEvNT_6ParamsEˇ ÄÄ*Ä2
8åI@åIHåIXb9gradient_tape/model/conv2d_13/Conv2D/Conv2DBackpropFilterh
ı
ï_ZN13cutlass_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi64ELi256ELi32EEENS4_51Conv2dWgradOutputGradientTileAccessIteratorAnalyticINS_11MatrixShapeILi64ELi32EEENS_6half_tENS_9transform29PitchLinearWarpRakedThreadMapINS_6layout16PitchLinearShapeILi64ELi32EEELi128ENSG_ILi8ELi4EEELi8EEEEENSD_11threadblock25RegularTileAccessIteratorISB_SC_NSF_40ColumnMajorTensorOpMultiplicandCongruousILi16ELi64EEELi1ESJ_Li16EEELNS_4arch14CacheOperation4KindE0ENS4_47Conv2dWgradActivationTileAccessIteratorAnalyticINSA_ILi32ELi256EEESC_NSE_INSG_ILi256ELi32EEELi128ESI_Li8EEEEENSM_ISU_SC_NSF_37RowMajorTensorOpMultiplicandCongruousILi16ELi64EEELi0ESW_Li16EEELSS_0ENS6_11threadblock9MmaPolicyINS6_4warp11MmaTensorOpINS7_ILi64ELi64ELi32EEESC_SO_SC_SZ_fNSF_8RowMajorENS13_17MmaTensorOpPolicyINSQ_3MmaINS7_ILi16ELi8ELi16EEELi32ESC_S16_SC_NSF_11ColumnMajorEfS16_NSQ_13OpMultiplyAddEEENSA_ILi1ELi1EEEEELi1ELb0EbEENSA_ILi0ELi0EEES1G_Li1EEELi4EbEENS_8epilogue11threadblock8EpilogueIS8_S1F_Li1ENS1K_22PredicatedTileIteratorINS1K_26OutputTileOptimalThreadMapINS1K_15OutputTileShapeILi256ELi8ELi1ELi1ELi1EEENS1O_ILi1ELi8ELi1ELi1ELi8EEELi128ELi4ELi32EEEfEENS1J_4warp24FragmentIteratorTensorOpIS15_S19_fNS_5ArrayIfLi4ELb1EEES16_EENS1T_20TileIteratorTensorOpIS15_S19_fS16_EENS1K_18SharedLoadIteratorINS1R_18CompactedThreadMapEfLi16EEENS1J_6thread17LinearCombinationIfLi4EffLNS_15FloatRoundStyleE2EEENSA_ILi0ELi8EEEEENS11_30GemmIdentityThreadblockSwizzleILi4EEELNS1_8OperatorE2ENS1_17Conv2dProblemSizeEEEEEvNT_6ParamsEˇ ÄÄ*Ä2	8å–I@å–IHå–IXb9gradient_tape/model/conv2d_20/Conv2D/Conv2DBackpropFilterh
¿
ﬁvoid xmma_cudnn::gemm::kernel<xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, xmma_cudnn::Col, 256, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 3> >(xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, xmma_cudnn::Col, 256, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 3>::Params)Ó Ä¿*Ä2-8åàH@åàHHåàHPXb9gradient_tape/model/conv2d_19/Conv2D/Conv2DBackpropFilterh
ñ
∂void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride>(cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride::Params)ı Ä¿*Ä2Ä8ãÿE@ãÿEHãÿEXb8gradient_tape/model/conv2d_13/Conv2D/Conv2DBackpropInputh
Ÿ
úvoid cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3>(cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3::Params)˛ Ä¿*Ä2Ä8ãòD@ãòDHãòDXbmodel/conv2d_7/Conv2Dh
Ø
Œvoid xmma_cudnn::gemm::kernel<xmma_cudnn::implicit_gemm::dgrad::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_a_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 16, false, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 16, xmma_cudnn::Row, 32, 256> >, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_c_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, 16, xmma_cudnn::Fragment_c<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, false>, false>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 3> >(xmma_cudnn::implicit_gemm::dgrad::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_a_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 16, false, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 16, xmma_cudnn::Row, 32, 256> >, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_c_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, 16, xmma_cudnn::Fragment_c<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, false>, false>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 3>::Params)ˇ Ä‡*Ä2Ä8ã¯C@ã¯CHã¯CPXb7gradient_tape/model/conv2d_7/Conv2D/Conv2DBackpropInputh
∫
|sm80_xmma_fprop_implicit_gemm_f16f16_f16f32_f32_nhwckrsc_nhwc_tilesize256x64x32_stage3_warpsize4x1x1_g1_tensor16x8x16_kernel˙ Ä‡*Ä2 8ãC@ãCHãCPXbmodel/conv2d_20/Conv2Dh
ø
ﬁvoid xmma_cudnn::gemm::kernel<xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, xmma_cudnn::Col, 256, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 3> >(xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, xmma_cudnn::Col, 256, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 3>::Params)Ó Ä¿*Ä2		8ä∞A@ä∞AHä∞APXb8gradient_tape/model/conv2d_7/Conv2D/Conv2DBackpropFilterh
ï
∂void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride>(cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride::Params)ı Ä¿*Ä2@8â:@â:Hâ:Xb8gradient_tape/model/conv2d_19/Conv2D/Conv2DBackpropInputh
ç
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2ÄÄ8â∞:@É∞HÉ»b>gradient_tape/model/batch_normalization_2/FusedBatchNormGradV3hu  »B
ç
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2ÄÄ8â∞:@É®HÉ»b>gradient_tape/model/batch_normalization_3/FusedBatchNormGradV3hu  »B
ç
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2ÄÄ8âò:@É∞HÉ∏b>gradient_tape/model/batch_normalization_1/FusedBatchNormGradV3hu  »B
Í
Évoid cudnn::bn_bw_1C11_kernel_new<__half, float, float2, 512, true, 1>(float, float, float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, float const*, float*, float*, float const*, float const*, float)(ê*Ä2Ä8â–8@â–8Hâ–8b>gradient_tape/model/batch_normalization_1/FusedBatchNormGradV3hu  »B
Í
Évoid cudnn::bn_bw_1C11_kernel_new<__half, float, float2, 512, true, 1>(float, float, float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, float const*, float*, float*, float const*, float const*, float)(ê*Ä2Ä8â∏7@â∏7Hâ∏7b>gradient_tape/model/batch_normalization_3/FusedBatchNormGradV3hu  »B
Æ
Œvoid cudnn::ops::pooling_bw_kernel_max<__half, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, false>(cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor) Ä *Ä2† 8â∞7@â∞7Hâ∞7b7gradient_tape/model/max_pooling2d_2/MaxPool/MaxPoolGradhu  »B
Í
Évoid cudnn::bn_bw_1C11_kernel_new<__half, float, float2, 512, true, 1>(float, float, float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, float const*, float*, float*, float const*, float const*, float)(ê*Ä2Ä8âê6@âê6Hâê6b>gradient_tape/model/batch_normalization_2/FusedBatchNormGradV3hu  »B
⁄
úvoid cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3>(cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3::Params)˛ Ä¿*Ä2Ä8á†1@á†1Há†1Xbmodel/conv2d_19/Conv2Dh
y
-ampere_fp16_s1688gemm_fp16_256x64_ldg8_f2f_ntÍÄ¿*Ä2Ä8á-@á-Há-b"gradient_tape/model/dense/MatMul_1hugUÖA
Ö
:ampere_fp16_s16816gemm_fp16_128x64_ldg8_f2f_stages_64x4_tnûÄÄ ÄÄ*Ä2Ä	8á†-@á†-Há†-Xb gradient_tape/model/dense/MatMulh
v
:ampere_fp16_s16816gemm_fp16_128x64_ldg8_f2f_stages_64x4_nnñÄÄ ÄÄ*Ä28á‡,@á‡,Há‡,Xbmodel/dense/MatMulh
Æ
Œvoid cudnn::ops::pooling_bw_kernel_max<__half, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, false>(cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor) Ä *Ä2† 8áË)@áË)HáË)b7gradient_tape/model/max_pooling2d_3/MaxPool/MaxPoolGradhu  »B
˚
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2ÄÄ8Üê)@É∏HÉÿb,model/batch_normalization_1/FusedBatchNormV3hu  »B
˚
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2ÄÄ8Üò'@É»HÉ–b,model/batch_normalization_2/FusedBatchNormV3hu  »B
˚
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2ÄÄ8áÄ'@É∞HÑ–b,model/batch_normalization_3/FusedBatchNormV3hu  »B
Ÿ

ü
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ü%@Ü%HÜ%bAdam/gradients/AddN_8huZUÖB
≠
ÿvoid cudnn::bn_fw_tr_1C11_kernel_NCHW<__half, float, 512, true, 1>(cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(ê*Ä2Ä8Ü∞ @Ü∞ HÜ∞ b,model/batch_normalization_2/FusedBatchNormV3hu  »B
≠
ÿvoid cudnn::bn_fw_tr_1C11_kernel_NCHW<__half, float, 512, true, 1>(cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(ê*Ä2Ä8Ö® @Ö® HÖ® b,model/batch_normalization_3/FusedBatchNormV3hu  »B
›
ùvoid convolve_common_engine_float_NHWC<__half, __half, 128, 5, 5, 3, 3, 3, true, false, false, false, false>(int, int, int, __half const*, __half const*, int, __half*, conv_kernel_common_params, unsigned long long, unsigned long, float, float, bool, __half const*, __half const*, bool)PÄ*2ÄÄ8Ö‡@Ö‡HÖ‡Xbmodel/conv2d_5/Conv2Dhu  HB
Ù
ï_ZN13cutlass_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi64ELi256ELi32EEENS4_51Conv2dWgradOutputGradientTileAccessIteratorAnalyticINS_11MatrixShapeILi64ELi32EEENS_6half_tENS_9transform29PitchLinearWarpRakedThreadMapINS_6layout16PitchLinearShapeILi64ELi32EEELi128ENSG_ILi8ELi4EEELi8EEEEENSD_11threadblock25RegularTileAccessIteratorISB_SC_NSF_40ColumnMajorTensorOpMultiplicandCongruousILi16ELi64EEELi1ESJ_Li16EEELNS_4arch14CacheOperation4KindE0ENS4_47Conv2dWgradActivationTileAccessIteratorAnalyticINSA_ILi32ELi256EEESC_NSE_INSG_ILi256ELi32EEELi128ESI_Li8EEEEENSM_ISU_SC_NSF_37RowMajorTensorOpMultiplicandCongruousILi16ELi64EEELi0ESW_Li16EEELSS_0ENS6_11threadblock9MmaPolicyINS6_4warp11MmaTensorOpINS7_ILi64ELi64ELi32EEESC_SO_SC_SZ_fNSF_8RowMajorENS13_17MmaTensorOpPolicyINSQ_3MmaINS7_ILi16ELi8ELi16EEELi32ESC_S16_SC_NSF_11ColumnMajorEfS16_NSQ_13OpMultiplyAddEEENSA_ILi1ELi1EEEEELi1ELb0EbEENSA_ILi0ELi0EEES1G_Li1EEELi4EbEENS_8epilogue11threadblock8EpilogueIS8_S1F_Li1ENS1K_22PredicatedTileIteratorINS1K_26OutputTileOptimalThreadMapINS1K_15OutputTileShapeILi256ELi8ELi1ELi1ELi1EEENS1O_ILi1ELi8ELi1ELi1ELi8EEELi128ELi4ELi32EEEfEENS1J_4warp24FragmentIteratorTensorOpIS15_S19_fNS_5ArrayIfLi4ELb1EEES16_EENS1T_20TileIteratorTensorOpIS15_S19_fS16_EENS1K_18SharedLoadIteratorINS1R_18CompactedThreadMapEfLi16EEENS1J_6thread17LinearCombinationIfLi4EffLNS_15FloatRoundStyleE2EEENSA_ILi0ELi8EEEEENS11_30GemmIdentityThreadblockSwizzleILi4EEELNS1_8OperatorE2ENS1_17Conv2dProblemSizeEEEEEvNT_6ParamsEˇ ÄÄ*Ä2)8Ö®@Ö®HÖ®Xb8gradient_tape/model/conv2d_5/Conv2D/Conv2DBackpropFilterh
≠
ÿvoid cudnn::bn_fw_tr_1C11_kernel_NCHW<__half, float, 512, true, 1>(cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(ê*Ä2Ä8Ö@ÖHÖb,model/batch_normalization_1/FusedBatchNormV3hu  »B
Ç
√void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*Ä2{8Öÿ@Å†HÉêbmodel/concatenate_1/concathuZUÖB
∑
htensorflow::functor::ReluGradHalfKernelVector(Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2ÄÄ8ÑÄ@ÑÄHÑÄb)gradient_tape/model/activation_2/ReluGradhu  »B
∑
htensorflow::functor::ReluGradHalfKernelVector(Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2ÄÄ8Ö¯@Ö¯HÖ¯b)gradient_tape/model/activation_1/ReluGradhu  »B
∑
htensorflow::functor::ReluGradHalfKernelVector(Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2ÄÄ8Ö¯@Ö¯HÖ¯b)gradient_tape/model/activation_3/ReluGradhu  »B
Ç
√void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*Ä2{8Ñ–@Å†HÇêbmodel/concatenate_3/concathuZUÖB
Ç
√void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*Ä2{8Ñ»@ÅòHÇêbmodel/concatenate_2/concathuZUÖB
ñ
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8Ñ¯@Ñ¯HÑ¯b<cond_1/then/_10/cond_1/Adam/Adam/update_50/ResourceApplyAdamhuZUÖB
ù
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_tn‚ÄÄ ÄÄ*Ä2Ä8É¿@É¿HÉ¿Xb7gradient_tape/model/conv2d_9/Conv2D/Conv2DBackpropInputh
å
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2ú8Éê@ÉêHÉêbAll_56hu  »B
{
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_nnﬁÄÄ ÄÄ*Ä2@8É¿@É¿HÉ¿Xbmodel/conv2d_21/Conv2Dh
£	
Ávoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*Ä2{8É®@É®HÉ®bmodel/activation_2/ReluhuZUÖB
£	
Ávoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ñ†@Ñ†HÑ†bmodel/activation_1/ReluhuZUÖB
£	
Ávoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*Ä2{8É†@É†HÉ†bmodel/activation_3/ReluhuZUÖB
ç
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2Ä†8É¿@ÅòHÅÄb>gradient_tape/model/batch_normalization_4/FusedBatchNormGradV3hu  »B
™
Àvoid cutlass_cudnn::Kernel<cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4> >(cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4>::Params)6* 2 28Ñ∏@Ñ∏HÑ∏Xb9gradient_tape/model/conv2d_11/Conv2D/Conv2DBackpropFilterhu  ñB
ù
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_tn‚ÄÄ ÄÄ*Ä2
@8Éê@ÉêHÉêXb8gradient_tape/model/conv2d_21/Conv2D/Conv2DBackpropInputh
ç
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2Ä†8É»@ÅòHÅòb>gradient_tape/model/batch_normalization_5/FusedBatchNormGradV3hu  »B
û
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_64x3_ntÊÄÄ ÄÄ*Ä2
8É»@É»HÉ»Xb9gradient_tape/model/conv2d_21/Conv2D/Conv2DBackpropFilterh
ñ
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8É»@É»HÉ»b<cond_1/then/_10/cond_1/Adam/Adam/update_52/ResourceApplyAdamhuZUÖB
{
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_nnﬁÄÄ ÄÄ*Ä2Ä8É‡@É‡HÉ‡Xbmodel/conv2d_9/Conv2Dh
ç
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2Äê8É‡@Å¿HÅÿb>gradient_tape/model/batch_normalization_7/FusedBatchNormGradV3hu  »B
ó
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Çà@ÇàHÇàb)gradient_tape/model/concatenate_1/Slice_2huZUÖB
ù
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_64x3_ntÊÄÄ ÄÄ*Ä28É@ÉHÉXb8gradient_tape/model/conv2d_9/Conv2D/Conv2DBackpropFilterh
ó
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Ç@ÇHÇb)gradient_tape/model/concatenate_2/Slice_1huZUÖB
ó
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Ç@ÇHÇb)gradient_tape/model/concatenate_3/Slice_1huZUÖB
Í
Évoid cudnn::bn_bw_1C11_kernel_new<__half, float, float2, 128, true, 1>(float, float, float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, float const*, float*, float*, float const*, float const*, float)(ê*Ä2Ä8Ç@ÇHÇb>gradient_tape/model/batch_normalization_4/FusedBatchNormGradV3hu  »B
Í
Évoid cudnn::bn_bw_1C11_kernel_new<__half, float, float2, 128, true, 1>(float, float, float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, float const*, float*, float*, float const*, float const*, float)(ê*Ä2Ä8Ç@ÇHÇb>gradient_tape/model/batch_normalization_5/FusedBatchNormGradV3hu  »B
¶
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Ç∞@Ç∞HÇ∞bmodel/conv2d_5/BiasAddhuZUÖB
ß
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Ç®@Ç®HÇ®bmodel/conv2d_10/BiasAddhuZUÖB
¶
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Ç†@Ç†HÇ†bmodel/conv2d_7/BiasAddhuZUÖB
–
åvoid pooling_fw_kernel_max_nhwc<__half, float, 0, (cudnnNanPropagation_t)0, false>(PoolingParams, __half*, __half const*, float, float, int)*Ä2ÄÄ8Ç®@Ç®HÇ®bmodel/max_pooling2d_1/MaxPoolhu  »B
Œ
åvoid pooling_fw_kernel_max_nhwc<__half, float, 0, (cudnnNanPropagation_t)0, false>(PoolingParams, __half*, __half const*, float, float, int)*Ä2ÄÄ8Çà@ÇàHÇàbmodel/max_pooling2d/MaxPoolhu  »B
™
Àvoid cutlass_cudnn::Kernel<cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4> >(cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4>::Params)6* 2@}8Ç¯@Ç¯HÇ¯Xb9gradient_tape/model/conv2d_23/Conv2D/Conv2DBackpropFilterhu  ñB
Õ
Ô_ZN13cutlass_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi64ELi64ELi32EEENS4_51Conv2dWgradOutputGradientTileAccessIteratorAnalyticINS_11MatrixShapeILi64ELi32EEENS_6half_tENS_9transform29PitchLinearWarpRakedThreadMapINS_6layout16PitchLinearShapeILi64ELi32EEELi128ENSG_ILi8ELi4EEELi8EEEEENSD_11threadblock25RegularTileAccessIteratorISB_SC_NSF_40ColumnMajorTensorOpMultiplicandCongruousILi16ELi64EEELi1ESJ_Li16EEELNS_4arch14CacheOperation4KindE0ENS4_47Conv2dWgradActivationTileAccessIteratorAnalyticINSA_ILi32ELi64EEESC_SJ_EENSM_ISU_SC_NSF_37RowMajorTensorOpMultiplicandCongruousILi16ELi64EEELi0ESJ_Li16EEELSS_0ENS6_11threadblock9MmaPolicyINS6_4warp11MmaTensorOpINS7_ILi32ELi32ELi32EEESC_SO_SC_SX_fNSF_8RowMajorENS11_17MmaTensorOpPolicyINSQ_3MmaINS7_ILi16ELi8ELi16EEELi32ESC_S14_SC_NSF_11ColumnMajorEfS14_NSQ_13OpMultiplyAddEEENSA_ILi1ELi1EEEEELi1ELb0EbEENSA_ILi0ELi0EEES1E_Li1EEELi10EbEENS_8epilogue11threadblock8EpilogueIS8_S1D_Li1ENS1I_22PredicatedTileIteratorINS1I_26OutputTileOptimalThreadMapINS1I_15OutputTileShapeILi64ELi8ELi2ELi1ELi1EEENS1M_ILi1ELi4ELi1ELi1ELi4EEELi128ELi4ELi32EEEfEENS1H_4warp24FragmentIteratorTensorOpIS13_S17_fNS_5ArrayIfLi4ELb1EEES14_EENS1R_20TileIteratorTensorOpIS13_S17_fS14_EENS1I_18SharedLoadIteratorINS1P_18CompactedThreadMapEfLi16EEENS1H_6thread17LinearCombinationIfLi4EffLNS_15FloatRoundStyleE2EEENSA_ILi0ELi8EEEEENSZ_30GemmIdentityThreadblockSwizzleILi4EEELNS1_8OperatorE2ENS1_17Conv2dProblemSizeEEEEEvNT_6ParamsEh ÄÄ*Ä2)8ÇË@ÇËHÇËXb8gradient_tape/model/conv2d_4/Conv2D/Conv2DBackpropFilterh
˚
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2Ä†8Ç»@Å†HÅ®b,model/batch_normalization_5/FusedBatchNormV3hu  »B
™
Àvoid cutlass_cudnn::Kernel<cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4> >(cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4>::Params)6* 2@?8Ç¿@Ç¿HÇ¿Xb9gradient_tape/model/conv2d_17/Conv2D/Conv2DBackpropFilterhu  ñB
˚
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2Ä†8Ç¿@Å†HÅ†b,model/batch_normalization_4/FusedBatchNormV3hu  »B
Ÿ

ü
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å¯@Å¯HÅ¯bAdam/gradients/AddN_6huZUÖB
™
Àvoid cutlass_cudnn::Kernel<cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4> >(cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4>::Params)6* 2@?8Ç–@Ç–HÇ–Xb9gradient_tape/model/conv2d_20/Conv2D/Conv2DBackpropFilterhu  ñB
˚
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2Äê8Ç†@Å–HÅ–b,model/batch_normalization_7/FusedBatchNormV3hu  »B
™
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å¿
@Å¿
HÅ¿
bAdam/gradients/AddN_29huZUÖB
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2¿8Ç®
@Ç®
HÇ®
b0gradient_tape/conv2d_22/kernel/Regularizer/Mul_1huZUÖB
õ
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Ç®
@Ç®
HÇ®
b0gradient_tape/model/conv2d_5/BiasAdd/BiasAddGradhuZUÖB
û
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_64x3_ntÊÄÄ ÄÄ*Ä28Ç†
@Ç†
HÇ†
Xb9gradient_tape/model/conv2d_15/Conv2D/Conv2DBackpropFilterh
˜
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ç†
@Ç†
HÇ†
Xb9gradient_tape/model/conv2d_22/Conv2D/Conv2DBackpropFilterhuZUÖB
û
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_64x3_ntÊÄÄ ÄÄ*Ä28Åò
@Åò
HÅò
Xb9gradient_tape/model/conv2d_12/Conv2D/Conv2DBackpropFilterh
™
Àvoid cutlass_cudnn::Kernel<cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4> >(cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4>::Params)6* 2@28ÇÄ
@ÇÄ
HÇÄ
Xb9gradient_tape/model/conv2d_14/Conv2D/Conv2DBackpropFilterhu  ñB
õ
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Ç¯	@Ç¯	HÇ¯	b0gradient_tape/model/conv2d_3/BiasAdd/BiasAddGradhuZUÖB
ú
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Ç¯	@Ç¯	HÇ¯	b1gradient_tape/model/conv2d_10/BiasAdd/BiasAddGradhuZUÖB
õ
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Ç¯	@Ç¯	HÇ¯	b0gradient_tape/model/conv2d_7/BiasAdd/BiasAddGradhuZUÖB
õ
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Ç	@Ç	HÇ	b0gradient_tape/model/conv2d_4/BiasAdd/BiasAddGradhuZUÖB
å
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2ÄP8Ç–	@ÅòHÄ†b>gradient_tape/model/batch_normalization_6/FusedBatchNormGradV3hu  »B
Ÿ

ü
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ç»	@Ç»	HÇ»	bAdam/gradients/AddN_7huZUÖB
Ÿ

ü
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å¿	@Å¿	HÅ¿	bAdam/gradients/AddN_9huZUÖB
©
Nvoid cudnn::ops::scalePackedTensor_kernel<__half, float>(long, __half*, float)*Ä2ˇˇ8Ç®	@Ç®	HÇ®	b5gradient_tape/model/max_pooling2d/MaxPool/MaxPoolGradhu  »B
‘
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ç†	@Ç†	HÇ†	Xbmodel/conv2d_22/Conv2DhuZUÖB
Ç
√void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*Ä2{8Å†	@Ä¯HÄÿbmodel/concatenate_4/concathuZUÖB
Ç
√void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*Ä2{8Å†	@Ä¯HÄÿbmodel/concatenate_5/concathuZUÖB
´
Nvoid cudnn::ops::scalePackedTensor_kernel<__half, float>(long, __half*, float)*Ä2ˇˇ8Å†	@Å†	HÅ†	b7gradient_tape/model/max_pooling2d_1/MaxPool/MaxPoolGradhu  »B
ˆ
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Åò	@Åò	HÅò	Xb8gradient_tape/model/conv2d_22/Conv2D/Conv2DBackpropInputhuZUÖB
∂
htensorflow::functor::ReluGradHalfKernelVector(Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2Ä(8Çê	@Çê	HÇê	b)gradient_tape/model/activation_4/ReluGradhu  »B
›
ùvoid convolve_common_engine_float_NHWC<__half, __half, 128, 5, 5, 3, 3, 3, true, false, false, false, false>(int, int, int, __half const*, __half const*, int, __half*, conv_kernel_common_params, unsigned long long, unsigned long, float, float, bool, __half const*, __half const*, bool)PÄ*2ÄÄ8Çê	@Çê	HÇê	Xbmodel/conv2d_4/Conv2Dhu  HB
∂
htensorflow::functor::ReluGradHalfKernelVector(Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2Ä(8Çà	@Çà	HÇà	b)gradient_tape/model/activation_5/ReluGradhu  »B
ñ
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8Ç¯@Ç¯HÇ¯b<cond_1/then/_10/cond_1/Adam/Adam/update_60/ResourceApplyAdamhuZUÖB
ñ
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8Ç‡@Ç‡HÇ‡b<cond_1/then/_10/cond_1/Adam/Adam/update_44/ResourceApplyAdamhuZUÖB
¸
ívoid cudnn::bn_bw_1C11_singleread_specialized<__half2, 512, 1, 2, 7>(float, float, float, float, cudnnTensorStruct, __half2 const*, cudnnTensorStruct, __half2 const*, cudnnTensorStruct, __half2*, float const*, float*, float*, float const*, float const*, float, cudnn::reduced_divisor, int, cudnn::reduced_divisor, cudnn::bnBwPersistentState*, int, float, float, float, int, float, cudnnStatus_t*, bool)8ê Ä *Ä2Ä8Ç–@Ç–HÇ–b>gradient_tape/model/batch_normalization_7/FusedBatchNormGradV3huZUÖB
ñ
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8Ç¿@Ç¿HÇ¿b<cond_1/then/_10/cond_1/Adam/Adam/update_36/ResourceApplyAdamhuZUÖB
◊
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2{8Å∏@Å∏HÅ∏b/gradient_tape/conv2d_22/kernel/Regularizer/TilehuZUÖB
Ç
√void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*Ä2{8Ç®@ÅàHÅ–bmodel/concatenate_7/concathuZUÖB
∂
htensorflow::functor::ReluGradHalfKernelVector(Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2Ä$8Åò@ÅòHÅòb)gradient_tape/model/activation_7/ReluGradhu  »B
ó
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Çê@ÇêHÇêb)gradient_tape/model/concatenate_1/Slice_1huZUÖB
ó
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Çê@ÇêHÇêb)gradient_tape/model/concatenate_3/Slice_2huZUÖB
ï
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Åê@ÅêHÅêb'gradient_tape/model/concatenate_1/SlicehuZUÖB
ï
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Çà@ÇàHÇàb'gradient_tape/model/concatenate_2/SlicehuZUÖB
ó
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Åà@ÅàHÅàb)gradient_tape/model/concatenate_2/Slice_2huZUÖB
ï
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8ÇÄ@ÇÄHÇÄb'gradient_tape/model/concatenate_3/SlicehuZUÖB
û
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_tn‚ÄÄ ÄÄ*Ä2Ä8Å@ÅHÅXb8gradient_tape/model/conv2d_15/Conv2D/Conv2DBackpropInputh
`
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä 8Ç¿@Ç¿HÇ¿bmodel/dropout_1/dropout/Mul_1huZUÖB
l
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä 8Å∏@Å∏HÅ∏b)gradient_tape/model/dropout/dropout/Mul_1huZUÖB
n
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä 8Å∏@Å∏HÅ∏b+gradient_tape/model/dropout_1/dropout/Mul_1huZUÖB
^
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä 8Å∞@Å∞HÅ∞bmodel/dropout/dropout/Mul_1huZUÖB
™
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å∞@Å∞HÅ∞bAdam/gradients/AddN_30huZUÖB
¶
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Ç†@Ç†HÇ†bmodel/conv2d_9/BiasAddhuZUÖB
ß
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Å†@Å†HÅ†bmodel/conv2d_11/BiasAddhuZUÖB
¶
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Çò@ÇòHÇòbmodel/conv2d_8/BiasAddhuZUÖB
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2–8Åò@ÅòHÅòb0gradient_tape/conv2d_23/kernel/Regularizer/Mul_1huZUÖB
¶
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Åò@ÅòHÅòbmodel/conv2d_3/BiasAddhuZUÖB
¶
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Åò@ÅòHÅòbmodel/conv2d_6/BiasAddhuZUÖB
¶
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Çê@ÇêHÇêbmodel/conv2d_4/BiasAddhuZUÖB
ñ
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8Åê@ÅêHÅêb<cond_1/then/_10/cond_1/Adam/Adam/update_28/ResourceApplyAdamhuZUÖB
K
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2¿8Çà@ÇàHÇàbmul_52huZUÖB
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2¿8ÅÄ@ÅÄHÅÄb.gradient_tape/conv2d_22/kernel/Regularizer/MulhuZUÖB
k
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2¿8ÅÄ@ÅÄHÅÄb#conv2d_22/kernel/Regularizer/SquarehuZUÖB
‘
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Å‡@Å‡HÅ‡Xbmodel/conv2d_23/Conv2DhuZUÖB
ˆ
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ç–@Ç–HÇ–Xb8gradient_tape/model/conv2d_23/Conv2D/Conv2DBackpropInputhuZUÖB
Œ
ıvoid cudnn::bn_fw_tr_1C11_singleread_specialized<__half2, 512, 1, 2, 20>(cudnnTensorStruct, __half2 const*, cudnnTensorStruct, __half2*, float const*, float const*, float, float, float*, float*, float*, float*, float, float, cudnn::reduced_divisor, int, cudnn::reduced_divisor, cudnn::bnFwPersistentState*, int, float, float, float, int, float, float, cudnnStatus_t*, bool)@ê Ä¿*Ä2Ä8Å»@Å»HÅ»b,model/batch_normalization_5/FusedBatchNormV3huZUÖB
|
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_nnﬁÄÄ ÄÄ*Ä2Ä8Å¿@Å¿HÅ¿Xbmodel/conv2d_15/Conv2Dh
û
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_tn‚ÄÄ ÄÄ*Ä2Ä8Å¿@Å¿HÅ¿Xb8gradient_tape/model/conv2d_12/Conv2D/Conv2DBackpropInputh
Œ
ıvoid cudnn::bn_fw_tr_1C11_singleread_specialized<__half2, 512, 1, 2, 20>(cudnnTensorStruct, __half2 const*, cudnnTensorStruct, __half2*, float const*, float const*, float, float, float*, float*, float*, float*, float, float, cudnn::reduced_divisor, int, cudnn::reduced_divisor, cudnn::bnFwPersistentState*, int, float, float, float, int, float, float, cudnnStatus_t*, bool)@ê Ä¿*Ä2Ä8Å¿@Å¿HÅ¿b,model/batch_normalization_4/FusedBatchNormV3huZUÖB
˙
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2ÄP8Ç∏@ÅòHÅ†b,model/batch_normalization_6/FusedBatchNormV3hu  »B
{
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_nnﬁÄÄ ÄÄ*Ä2@8Å∞@Å∞HÅ∞Xbmodel/conv2d_18/Conv2Dh
Æ
Ìvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å∞@Å∞HÅ∞bmodel/dropout_1/dropout/CasthuZUÖB
£	
Ávoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å∞@Å∞HÅ∞bmodel/activation_4/ReluhuZUÖB
£	
Ávoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å∞@Å∞HÅ∞bmodel/activation_5/ReluhuZUÖB
Õ
Ùvoid cudnn::bn_fw_tr_1C11_singleread_specialized<__half2, 512, 1, 2, 0>(cudnnTensorStruct, __half2 const*, cudnnTensorStruct, __half2*, float const*, float const*, float, float, float*, float*, float*, float*, float, float, cudnn::reduced_divisor, int, cudnn::reduced_divisor, cudnn::bnFwPersistentState*, int, float, float, float, int, float, float, cudnnStatus_t*, bool)(ê ÄÄ*Ä2Ä8Å∞@Å∞HÅ∞b,model/batch_normalization_7/FusedBatchNormV3hu  »B
ñ
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8Å∞@Å∞HÅ∞b<cond_1/then/_10/cond_1/Adam/Adam/update_42/ResourceApplyAdamhuZUÖB
¨
Ìvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å®@Å®HÅ®bmodel/dropout/dropout/CasthuZUÖB
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å®@Å®HÅ®b.gradient_tape/model/conv2d_22/Conv2D/Cast/CasthuZUÖB
û
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_ntﬁÄÄ ÄÄ*Ä28Åò@ÅòHÅòXb9gradient_tape/model/conv2d_18/Conv2D/Conv2DBackpropFilterh
Ÿ

ü
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Åê@ÅêHÅêbAdam/gradients/AddN_4huZUÖB
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å@ÅHÅbmodel/conv2d_22/Conv2D/CasthuZUÖB
◊
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2{8Å@ÅHÅb/gradient_tape/conv2d_23/kernel/Regularizer/TilehuZUÖB
ì
1ampere_s16816gemm_fp16_128x64_ldg8_stages_64x4_ntíÄÄ ÄÄ*Ä28ÅË@ÅËHÅËXb8gradient_tape/model/conv2d_6/Conv2D/Conv2DBackpropFilterh
£	
Ávoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*Ä2{8ÄË@ÄËHÄËbmodel/activation_7/ReluhuZUÖB
ù
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_tn‚ÄÄ ÄÄ*Ä2@8Ä∞@Ä∞HÄ∞Xb8gradient_tape/model/conv2d_18/Conv2D/Conv2DBackpropInputh
o
.ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_nnˆÄÄ*Ä2Ä8Å®@Å®HÅ®Xbmodel/conv2d_6/Conv2DhugUÖA
õ
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Å®@Å®HÅ®b0gradient_tape/model/conv2d_6/BiasAdd/BiasAddGradhuZUÖB
ë
.ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_tn˛ÄÄ*Ä2Ä8Å†@Å†HÅ†Xb7gradient_tape/model/conv2d_6/Conv2D/Conv2DBackpropInputhugUÖA
õ
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Å†@Å†HÅ†b0gradient_tape/model/conv2d_8/BiasAdd/BiasAddGradhuZUÖB
õ
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Å†@Å†HÅ†b0gradient_tape/model/conv2d_9/BiasAdd/BiasAddGradhuZUÖB
|
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_nnﬁÄÄ ÄÄ*Ä2Ä8Ä†@Ä†HÄ†Xbmodel/conv2d_12/Conv2Dh
ú
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Åò@ÅòHÅòb1gradient_tape/model/conv2d_11/BiasAdd/BiasAddGradhuZUÖB
ú
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  28Åê@ÅêHÅêb1gradient_tape/model/conv2d_19/BiasAdd/BiasAddGradhuZUÖB
ú
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Åê@ÅêHÅêb1gradient_tape/model/conv2d_21/BiasAdd/BiasAddGradhuZUÖB
ú
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Äê@ÄêHÄêb1gradient_tape/model/conv2d_22/BiasAdd/BiasAddGradhuZUÖB
j
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä 8Åà@ÅàHÅàb'gradient_tape/model/dropout/dropout/MulhuZUÖB
l
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä 8Åà@ÅàHÅàb)gradient_tape/model/dropout_1/dropout/MulhuZUÖB
\
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä 8ÅÄ@ÅÄHÅÄbmodel/dropout/dropout/MulhuZUÖB
^
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä 8ÄÄ@ÄÄHÄÄbmodel/dropout_1/dropout/MulhuZUÖB
¸
ívoid cudnn::bn_bw_1C11_singleread_specialized<__half2, 512, 1, 2, 7>(float, float, float, float, cudnnTensorStruct, __half2 const*, cudnnTensorStruct, __half2 const*, cudnnTensorStruct, __half2*, float const*, float*, float*, float const*, float const*, float, cudnn::reduced_divisor, int, cudnn::reduced_divisor, cudnn::bnBwPersistentState*, int, float, float, float, int, float, cudnnStatus_t*, bool)8ê Ä *Ä2Ä
8ÄÄ@ÄÄHÄÄb>gradient_tape/model/batch_normalization_6/FusedBatchNormGradV3huZUÖB
ú
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  28ÄÄ@ÄÄHÄÄb1gradient_tape/model/conv2d_18/BiasAdd/BiasAddGradhuZUÖB
K
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2–8Å¯@Å¯HÅ¯bmul_54huZUÖB
Ç
√void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*Ä2{8Å¯@ÄàHÅ¯bmodel/concatenate_6/concathuZUÖB
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2–8Å@ÅHÅb.gradient_tape/conv2d_23/kernel/Regularizer/MulhuZUÖB
k
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2–8Ä@ÄHÄb#conv2d_23/kernel/Regularizer/SquarehuZUÖB
˜
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8ÅË@ÅËHÅËXb9gradient_tape/model/conv2d_23/Conv2D/Conv2DBackpropFilterhuZUÖB
∂
htensorflow::functor::ReluGradHalfKernelVector(Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2Ä8Å–@Å–HÅ–b)gradient_tape/model/activation_6/ReluGradhu  »B
™
Àvoid cutlass_cudnn::Kernel<cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4> >(cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4>::Params)6* 2@8Å»@Å»HÅ»Xb9gradient_tape/model/conv2d_16/Conv2D/Conv2DBackpropFilterhu  ñB
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å∞@Å∞HÅ∞b.gradient_tape/model/conv2d_23/Conv2D/Cast/CasthuZUÖB
–
åvoid pooling_fw_kernel_max_nhwc<__half, float, 0, (cudnnNanPropagation_t)0, false>(PoolingParams, __half*, __half const*, float, float, int)*Ä2Ä†8Ä∞@Ä∞HÄ∞bmodel/max_pooling2d_2/MaxPoolhu  »B
T
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2ÄZ8Å®@Å®HÅ®bIsFinite_50hu  »B
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Åò@ÅòHÅòbmodel/conv2d_23/Conv2D/CasthuZUÖB
ó
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Åê@ÅêHÅêb)gradient_tape/model/concatenate_4/Slice_2huZUÖB
ó
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Äê@ÄêHÄêb)gradient_tape/model/concatenate_4/Slice_1huZUÖB
ï
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Äê@ÄêHÄêb'gradient_tape/model/concatenate_7/SlicehuZUÖB
ó
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Åà@ÅàHÅàb)gradient_tape/model/concatenate_5/Slice_2huZUÖB
ó
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Åà@ÅàHÅàb)gradient_tape/model/concatenate_7/Slice_1huZUÖB
ó
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Äà@ÄàHÄàb)gradient_tape/model/concatenate_5/Slice_1huZUÖB
–
åvoid pooling_fw_kernel_max_nhwc<__half, float, 0, (cudnnNanPropagation_t)0, false>(PoolingParams, __half*, __half const*, float, float, int)*Ä2Äê8ÅÄ@ÅÄHÅÄbmodel/max_pooling2d_3/MaxPoolhu  »B
Ä
ßvoid tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>)*Ä2{8ÅÄ@ÅÄHÅÄb4model/dropout_1/dropout/random_uniform/RandomUniformhuZUÖB
˛
ßvoid tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>)*Ä2{8ÄÄ@ÄÄHÄÄb2model/dropout/dropout/random_uniform/RandomUniformhuZUÖB
n
'GreaterEqual_GPU_DT_HALF_DT_BOOL_kernel*Ä2Ä 8ÅË@ÅËHÅËb"model/dropout/dropout/GreaterEqualhuZUÖB
™
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8ÅË@ÅËHÅËbAdam/gradients/AddN_27huZUÖB
™
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8ÅË@ÅËHÅËbAdam/gradients/AddN_32huZUÖB
™
Àvoid cutlass_cudnn::Kernel<cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4> >(cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4>::Params)6* 2@8ÅË@ÅËHÅËXb9gradient_tape/model/conv2d_13/Conv2D/Conv2DBackpropFilterhu  ñB
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ä8ÄË@ÄËHÄËb.gradient_tape/dense_1/kernel/Regularizer/Mul_1huZUÖB
p
'GreaterEqual_GPU_DT_HALF_DT_BOOL_kernel*Ä2Ä 8Å‡@Å‡HÅ‡b$model/dropout_1/dropout/GreaterEqualhuZUÖB
ß
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Å‡@Å‡HÅ‡bmodel/conv2d_14/BiasAddhuZUÖB
ß
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Å‡@Å‡HÅ‡bmodel/conv2d_16/BiasAddhuZUÖB
ß
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Å‡@Å‡HÅ‡bmodel/conv2d_21/BiasAddhuZUÖB
™
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä‡@Ä‡HÄ‡bAdam/gradients/AddN_24huZUÖB
ß
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Ä‡@Ä‡HÄ‡bmodel/conv2d_22/BiasAddhuZUÖB
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ë8Åÿ@ÅÿHÅÿb0gradient_tape/conv2d_17/kernel/Regularizer/Mul_1huZUÖB
ß
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Åÿ@ÅÿHÅÿbmodel/conv2d_17/BiasAddhuZUÖB
ß
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Äÿ@ÄÿHÄÿbmodel/conv2d_13/BiasAddhuZUÖB
ñ
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8Äÿ@ÄÿHÄÿb<cond_1/then/_10/cond_1/Adam/Adam/update_20/ResourceApplyAdamhuZUÖB
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ë8Ä–@Ä–HÄ–b0gradient_tape/conv2d_20/kernel/Regularizer/Mul_1huZUÖB
Õ
Ùvoid cudnn::bn_fw_tr_1C11_singleread_specialized<__half2, 512, 1, 2, 0>(cudnnTensorStruct, __half2 const*, cudnnTensorStruct, __half2*, float const*, float const*, float, float, float*, float*, float*, float*, float, float, cudnn::reduced_divisor, int, cudnn::reduced_divisor, cudnn::bnFwPersistentState*, int, float, float, float, int, float, float, cudnnStatus_t*, bool)(ê ÄÄ*Ä2Ä
8Ä–@Ä–HÄ–b,model/batch_normalization_6/FusedBatchNormV3hu  »B
√
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2ú8Å»@Å»HÅ»b conv2d_22/kernel/Regularizer/Sumhu  »B
`
 ampere_fp16_sgemm_fp16_32x128_nn9ÄÄ*Ä2Ä 8Å¿@Å¿HÅ¿Xbmodel/conv2d_3/Conv2DhuZUÖB
‘
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ä∏@Ä∏HÄ∏Xbmodel/conv2d_20/Conv2DhuZUÖB
ˆ
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Å∞@Å∞HÅ∞Xb8gradient_tape/model/conv2d_17/Conv2D/Conv2DBackpropInputhuZUÖB
‘
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Å∞@Å∞HÅ∞Xbmodel/conv2d_17/Conv2DhuZUÖB
ˆ
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ä∞@Ä∞HÄ∞Xb8gradient_tape/model/conv2d_20/Conv2D/Conv2DBackpropInputhuZUÖB
â
§void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nt_align1>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nt_align1::Params)¢ Ä¿*Ä2¢8Å†@Å†HÅ†Xb8gradient_tape/model/conv2d_3/Conv2D/Conv2DBackpropFilterhugUÖA
£	
Ávoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä†@Ä†HÄ†bmodel/activation_6/ReluhuZUÖB
ñ
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8Ä†@Ä†HÄ†b<cond_1/then/_10/cond_1/Adam/Adam/update_34/ResourceApplyAdamhuZUÖB
Ÿ

ü
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Åò@ÅòHÅòbAdam/gradients/AddN_5huZUÖB
´
Nvoid cudnn::ops::scalePackedTensor_kernel<__half, float>(long, __half*, float)*Ä2ˇˇ8Åê@ÅêHÅêb7gradient_tape/model/max_pooling2d_2/MaxPool/MaxPoolGradhu  »B
™
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Åà@ÅàHÅàbAdam/gradients/AddN_21huZUÖB
◊
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2{8Åà@ÅàHÅàb/gradient_tape/conv2d_17/kernel/Regularizer/TilehuZUÖB
T
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2¿>8Äà@ÄàHÄàbIsFinite_52hu  »B
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8ÄÄ@ÄÄHÄÄb0gradient_tape/conv2d_14/kernel/Regularizer/Mul_1huZUÖB
K
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ë8Å¯@Å¯HÅ¯bmul_38huZUÖB
ñ
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8Ä¯@Ä¯HÄ¯b<cond_1/then/_10/cond_1/Adam/Adam/update_48/ResourceApplyAdamhuZUÖB
´
Nvoid cudnn::ops::scalePackedTensor_kernel<__half, float>(long, __half*, float)*Ä2ˇˇ8Ä@ÄHÄb7gradient_tape/model/max_pooling2d_3/MaxPool/MaxPoolGradhu  »B
ú
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8ÄË@ÄËHÄËb1gradient_tape/model/conv2d_13/BiasAdd/BiasAddGradhuZUÖB
ú
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8ÄË@ÄËHÄËb1gradient_tape/model/conv2d_16/BiasAdd/BiasAddGradhuZUÖB
™
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å‡@Å‡HÅ‡bAdam/gradients/AddN_26huZUÖB
ú
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Å‡@Å‡HÅ‡b1gradient_tape/model/conv2d_14/BiasAdd/BiasAddGradhuZUÖB
‘
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Å‡@Å‡HÅ‡Xbmodel/conv2d_14/Conv2DhuZUÖB
K
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ä8Ä‡@Ä‡HÄ‡bmul_62huZUÖB
ú
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Ä‡@Ä‡HÄ‡b1gradient_tape/model/conv2d_17/BiasAdd/BiasAddGradhuZUÖB
ˆ
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ä‡@Ä‡HÄ‡Xb8gradient_tape/model/conv2d_14/Conv2D/Conv2DBackpropInputhuZUÖB
ñ
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8Åÿ@ÅÿHÅÿb<cond_1/then/_10/cond_1/Adam/Adam/update_18/ResourceApplyAdamhuZUÖB
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2–8Äÿ@ÄÿHÄÿb0gradient_tape/conv2d_19/kernel/Regularizer/Mul_1huZUÖB
ñ
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8Äÿ@ÄÿHÄÿb<cond_1/then/_10/cond_1/Adam/Adam/update_26/ResourceApplyAdamhuZUÖB
q
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ä8Å–@Å–HÅ–b,gradient_tape/dense_1/kernel/Regularizer/MulhuZUÖB
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ë8Ä»@Ä»HÄ»b.gradient_tape/conv2d_17/kernel/Regularizer/MulhuZUÖB
K
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ë8Å¿@Å¿HÅ¿bmul_46huZUÖB
k
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ë8Å¿@Å¿HÅ¿b#conv2d_20/kernel/Regularizer/SquarehuZUÖB
ˆ
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Å¿@Å¿HÅ¿Xb8gradient_tape/model/conv2d_19/Conv2D/Conv2DBackpropInputhuZUÖB
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ë8Ä¿@Ä¿HÄ¿b.gradient_tape/conv2d_20/kernel/Regularizer/MulhuZUÖB
k
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ë8Ä¿@Ä¿HÄ¿b#conv2d_17/kernel/Regularizer/SquarehuZUÖB
i
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ä8Ä¿@Ä¿HÄ¿b!dense_1/kernel/Regularizer/SquarehuZUÖB
◊
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2{8Ä¿@Ä¿HÄ¿b/gradient_tape/conv2d_14/kernel/Regularizer/TilehuZUÖB
˜
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ä¿@Ä¿HÄ¿Xb9gradient_tape/model/conv2d_20/Conv2D/Conv2DBackpropFilterhuZUÖB
`
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä
8Å∏@Å∏HÅ∏bmodel/dropout_2/dropout/Mul_1huZUÖB
√
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2–8Å∏@Å∏HÅ∏b conv2d_23/kernel/Regularizer/Sumhu  »B
‘
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Å∏@Å∏HÅ∏Xbmodel/conv2d_19/Conv2DhuZUÖB
n
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä
8Ä∏@Ä∏HÄ∏b+gradient_tape/model/dropout_2/dropout/Mul_1huZUÖB
◊
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2{8Å∞@Å∞HÅ∞b/gradient_tape/conv2d_19/kernel/Regularizer/TilehuZUÖB
˜
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Å∞@Å∞HÅ∞Xb9gradient_tape/model/conv2d_17/Conv2D/Conv2DBackpropFilterhuZUÖB
¥
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä∞@Ä∞HÄ∞b,gradient_tape/model/dense_1/MatMul/Cast/CasthuZUÖB
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å®@Å®HÅ®b.gradient_tape/model/conv2d_17/Conv2D/Cast/CasthuZUÖB
n
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä	8Å†@Å†HÅ†b+gradient_tape/model/dropout_3/dropout/Mul_1huZUÖB
`
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä	8Å†@Å†HÅ†bmodel/dropout_3/dropout/Mul_1huZUÖB
≠
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å†@Å†HÅ†bmodel/dense_1/MatMul/CasthuZUÖB
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä†@Ä†HÄ†b.gradient_tape/model/conv2d_20/Conv2D/Cast/CasthuZUÖB
ï
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Åò@ÅòHÅòb'gradient_tape/model/concatenate_4/SlicehuZUÖB
ï
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Åò@ÅòHÅòb'gradient_tape/model/concatenate_5/SlicehuZUÖB
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Äò@ÄòHÄòbmodel/conv2d_17/Conv2D/CasthuZUÖB
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Äò@ÄòHÄòbmodel/conv2d_20/Conv2D/CasthuZUÖB
K
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8Åê@ÅêHÅêbmul_30huZUÖB
Æ
Ìvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Äê@ÄêHÄêbmodel/dropout_2/dropout/CasthuZUÖB
ï
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Äê@ÄêHÄêb'gradient_tape/model/concatenate_6/SlicehuZUÖB
ó
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Äê@ÄêHÄêb)gradient_tape/model/concatenate_6/Slice_1huZUÖB
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8Åà@ÅàHÅàb.gradient_tape/conv2d_14/kernel/Regularizer/MulhuZUÖB
k
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8ÄÄ@ÄÄHÄÄb#conv2d_14/kernel/Regularizer/SquarehuZUÖB
ß
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8ÄÄ@ÄÄHÄÄbmodel/conv2d_15/BiasAddhuZUÖB
Æ
Ìvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä¯@Ä¯HÄ¯bmodel/dropout_3/dropout/CasthuZUÖB
ß
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Ä¯@Ä¯HÄ¯bmodel/conv2d_18/BiasAddhuZUÖB
ß
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Ä¯@Ä¯HÄ¯bmodel/conv2d_19/BiasAddhuZUÖB
˜
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ä¯@Ä¯HÄ¯Xb9gradient_tape/model/conv2d_14/Conv2D/Conv2DBackpropFilterhuZUÖB
ß
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Å@ÅHÅbmodel/conv2d_12/BiasAddhuZUÖB
K
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2–8Ä@ÄHÄbmul_44huZUÖB
˜
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ä@ÄHÄXb9gradient_tape/model/conv2d_19/Conv2D/Conv2DBackpropFilterhuZUÖB
T
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2Ä 8ÅË@ÅËHÅËbIsFinite_60hu  »B
k
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2–8ÅË@ÅËHÅËb#conv2d_19/kernel/Regularizer/SquarehuZUÖB
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2–8ÄË@ÄËHÄËb.gradient_tape/conv2d_19/kernel/Regularizer/MulhuZUÖB
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8ÄË@ÄËHÄËbmodel/conv2d_14/Conv2D/CasthuZUÖB
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8ÄË@ÄËHÄËb.gradient_tape/model/conv2d_14/Conv2D/Cast/CasthuZUÖB
l
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä
8Äÿ@ÄÿHÄÿb)gradient_tape/model/dropout_2/dropout/MulhuZUÖB
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Äÿ@ÄÿHÄÿbmodel/conv2d_19/Conv2D/CasthuZUÖB
™
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å–@Å–HÅ–bAdam/gradients/AddN_18huZUÖB
T
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2†8Ä–@Ä–HÄ–bIsFinite_36hu  »B
T
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2†8Ä–@Ä–HÄ–bIsFinite_44hu  »B
^
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä
8Ä–@Ä–HÄ–bmodel/dropout_2/dropout/MulhuZUÖB
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä–@Ä–HÄ–b.gradient_tape/model/conv2d_19/Conv2D/Cast/CasthuZUÖB
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2ê8Å»@Å»HÅ»b0gradient_tape/conv2d_11/kernel/Regularizer/Mul_1huZUÖB
l
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä	8Å»@Å»HÅ»b)gradient_tape/model/dropout_3/dropout/MulhuZUÖB
’
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Ä»@Ä»HÄ»b-gradient_tape/dense_1/kernel/Regularizer/TilehuZUÖB
√
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2Ë8Ä¿@Ä¿HÄ¿b conv2d_20/kernel/Regularizer/Sumhu  »B
ú
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Ä¿@Ä¿HÄ¿b1gradient_tape/model/conv2d_15/BiasAdd/BiasAddGradhuZUÖB
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ë8Å∏@Å∏HÅ∏b0gradient_tape/conv2d_16/kernel/Regularizer/Mul_1huZUÖB
^
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä	8Ä∏@Ä∏HÄ∏bmodel/dropout_3/dropout/MulhuZUÖB
™
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä∏@Ä∏HÄ∏bAdam/gradients/AddN_23huZUÖB
ú
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Ä∏@Ä∏HÄ∏b1gradient_tape/model/conv2d_12/BiasAdd/BiasAddGradhuZUÖB
ˆ
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ä∏@Ä∏HÄ∏Xb8gradient_tape/model/conv2d_11/Conv2D/Conv2DBackpropInputhuZUÖB
‘
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ä∏@Ä∏HÄ∏Xbmodel/conv2d_11/Conv2DhuZUÖB
‘
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Å∞@Å∞HÅ∞Xbmodel/conv2d_16/Conv2DhuZUÖB
Å
Campere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x5_nníÄÄ ÄÄ*Ä2 8Ä∞@Ä∞HÄ∞Xbmodel/dense_1/MatMulh
è
Campere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tníÄÄ ÄÄ*Ä2 8Ä∞@Ä∞HÄ∞Xb"gradient_tape/model/dense_1/MatMulh
¡
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2Ä8Ä∞@Ä∞HÄ∞bdense_1/kernel/Regularizer/Sumhu  »B
Ä
ßvoid tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>)*Ä2{8Ä∞@Ä∞HÄ∞b4model/dropout_2/dropout/random_uniform/RandomUniformhuZUÖB
p
'GreaterEqual_GPU_DT_HALF_DT_BOOL_kernel*Ä2Ä
8Å®@Å®HÅ®b$model/dropout_2/dropout/GreaterEqualhuZUÖB
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2¿8Å®@Å®HÅ®b0gradient_tape/conv2d_21/kernel/Regularizer/Mul_1huZUÖB
™
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å®@Å®HÅ®bAdam/gradients/AddN_28huZUÖB
T
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2Ä8Ä®@Ä®HÄ®bIsFinite_28hu  »B
◊
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2{8Ä®@Ä®HÄ®b/gradient_tape/conv2d_11/kernel/Regularizer/TilehuZUÖB
√
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2Ë8Ä®@Ä®HÄ®b conv2d_17/kernel/Regularizer/Sumhu  »B
Ÿ
˙void nhwcAddPaddingKernel<__half, __half, float, true, (cudnnKernelDataType_t)0>(int, int, int, int, int, int, int, int, __half const*, __half*, int, int, int, int, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor)*Ä2§8Ä®@ÄHÄêXb8gradient_tape/model/conv2d_4/Conv2D/Conv2DBackpropFilterhu  »B
ˆ
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ä®@Ä®HÄ®Xb8gradient_tape/model/conv2d_16/Conv2D/Conv2DBackpropInputhuZUÖB
◊
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2{8Ä†@Ä†HÄ†b/gradient_tape/conv2d_16/kernel/Regularizer/TilehuZUÖB
Ÿ
˙void nhwcAddPaddingKernel<__half, __half, float, true, (cudnnKernelDataType_t)0>(int, int, int, int, int, int, int, int, __half const*, __half*, int, int, int, int, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor)*Ä2§8Ä†@ÄHÄàXb8gradient_tape/model/conv2d_5/Conv2D/Conv2DBackpropFilterhu  »B
Ä
ßvoid tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>)*Ä2{8Ä†@Ä†HÄ†b4model/dropout_3/dropout/random_uniform/RandomUniformhuZUÖB
p
'GreaterEqual_GPU_DT_HALF_DT_BOOL_kernel*Ä2Ä	8Åò@ÅòHÅòb$model/dropout_3/dropout/GreaterEqualhuZUÖB
™
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Åò@ÅòHÅòbAdam/gradients/AddN_17huZUÖB
T
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2¿8Äò@ÄòHÄòbIsFinite_42hu  »B
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8Äò@ÄòHÄòb0gradient_tape/conv2d_13/kernel/Regularizer/Mul_1huZUÖB
ó
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Äò@ÄòHÄòb)gradient_tape/model/concatenate_7/Slice_2huZUÖB
™
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Äò@ÄòHÄòbAdam/gradients/AddN_20huZUÖB
ó
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Åê@ÅêHÅêb)gradient_tape/model/concatenate_6/Slice_2huZUÖB
å
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2†8Åê@ÅêHÅêbAll_50hu  »B
‘
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Åê@ÅêHÅêXbmodel/conv2d_13/Conv2DhuZUÖB
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8Äê@ÄêHÄêb0gradient_tape/conv2d_10/kernel/Regularizer/Mul_1huZUÖB
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2ê8Äê@ÄêHÄêb.gradient_tape/conv2d_11/kernel/Regularizer/MulhuZUÖB
K
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2ê8Äê@ÄêHÄêbmul_22huZUÖB
k
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2ê8Äê@ÄêHÄêb#conv2d_11/kernel/Regularizer/SquarehuZUÖB
ˆ
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Äê@ÄêHÄêXb8gradient_tape/model/conv2d_10/Conv2D/Conv2DBackpropInputhuZUÖB
ˆ
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Äê@ÄêHÄêXb8gradient_tape/model/conv2d_13/Conv2D/Conv2DBackpropInputhuZUÖB
‘
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Äê@ÄêHÄêXbmodel/conv2d_10/Conv2DhuZUÖB
˜
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Äê@ÄêHÄêXb9gradient_tape/model/conv2d_11/Conv2D/Conv2DBackpropFilterhuZUÖB
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ë8Åà@ÅàHÅàb.gradient_tape/conv2d_16/kernel/Regularizer/MulhuZUÖB
ß
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Åà@ÅàHÅàbmodel/conv2d_20/BiasAddhuZUÖB
◊
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2{8Äà@ÄàHÄàb/gradient_tape/conv2d_10/kernel/Regularizer/TilehuZUÖB
◊
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2{8Äà@ÄàHÄàb/gradient_tape/conv2d_21/kernel/Regularizer/TilehuZUÖB
√
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2†8Äà@ÄàHÄàb conv2d_14/kernel/Regularizer/Sumhu  »B
ñ
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8Äà@ÄàHÄàb<cond_1/then/_10/cond_1/Adam/Adam/update_12/ResourceApplyAdamhuZUÖB
K
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ë8ÄÄ@ÄÄHÄÄbmul_36huZUÖB
k
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ë8ÄÄ@ÄÄHÄÄb#conv2d_16/kernel/Regularizer/SquarehuZUÖB
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8ÄÄ@ÄÄHÄÄbmodel/conv2d_11/Conv2D/CasthuZUÖB
√
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2–8ÄÄ@ÄÄHÄÄb conv2d_19/kernel/Regularizer/Sumhu  »B
ß
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8ÄÄ@ÄÄHÄÄbmodel/conv2d_23/BiasAddhuZUÖB
ú
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8ÄÄ@ÄÄHÄÄb1gradient_tape/model/conv2d_20/BiasAdd/BiasAddGradhuZUÖB
ú
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8ÄÄ@ÄÄHÄÄb1gradient_tape/model/conv2d_23/BiasAdd/BiasAddGradhuZUÖB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2¿8Äx@ÄxHÄxb.gradient_tape/conv2d_21/kernel/Regularizer/MulhuZUÖB
H
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2¿8Äx@ÄxHÄxbmul_50huZUÖB
§
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Äx@ÄxHÄxbmodel/conv2d_3/CasthuZUÖB
≥
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Äx@ÄxHÄxb.gradient_tape/model/conv2d_11/Conv2D/Cast/CasthuZUÖB
ì
Ávoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<unsigned char const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<unsigned char const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Äx@ÄxHÄxb
model/CasthuZUÖB
Ù
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Äx@ÄxHÄxXb9gradient_tape/model/conv2d_16/Conv2D/Conv2DBackpropFilterhuZUÖB
â
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2Ë8Åp@ÅpHÅpbAll_52hu  »B
h
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2¿8Äp@ÄpHÄpb#conv2d_21/kernel/Regularizer/SquarehuZUÖB
¨
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Äp@ÄpHÄpbmodel/conv2d_16/Conv2D/CasthuZUÖB
Ì
ëvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorForcedEvalOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorForcedEvalOp<Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_log_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorForcedEvalOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorForcedEvalOp<Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_log_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long)2*Ä28Äp@ÄpHÄpb:categorical_crossentropy/softmax_cross_entropy_with_logitshuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8Äp@ÄpHÄpb<cond_1/then/_10/cond_1/Adam/Adam/update_40/ResourceApplyAdamhuZUÖB
Ñ
:ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_ntÍÄÄ*Ä28Åh@ÅhHÅhb$gradient_tape/model/dense_1/MatMul_1hugUÖA
≥
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Åh@ÅhHÅhb.gradient_tape/model/conv2d_16/Conv2D/Cast/CasthuZUÖB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8Äh@ÄhHÄhb.gradient_tape/conv2d_10/kernel/Regularizer/MulhuZUÖB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8Äh@ÄhHÄhb.gradient_tape/conv2d_13/kernel/Regularizer/MulhuZUÖB
H
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8Äh@ÄhHÄhbmul_20huZUÖB
H
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8Äh@ÄhHÄhbmul_28huZUÖB
h
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8Äh@ÄhHÄhb#conv2d_10/kernel/Regularizer/SquarehuZUÖB
h
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8Äh@ÄhHÄhb#conv2d_13/kernel/Regularizer/SquarehuZUÖB
¨
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Äh@ÄhHÄhbmodel/conv2d_21/Conv2D/CasthuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8Äh@ÄhHÄhb<cond_1/then/_10/cond_1/Adam/Adam/update_10/ResourceApplyAdamhuZUÖB
»
Êvoid xmma_cudnn::gemm::split_k_kernel<xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, xmma_cudnn::Col, 256, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 3> >(xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, xmma_cudnn::Col, 256, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 3>::Params)? Äê*Ä2	8Äh@ÄhHÄhPXb8gradient_tape/model/conv2d_7/Conv2D/Conv2DBackpropFilterhuMUB
Q
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2¿8Ä`@Ä`HÄ`bIsFinite_20hu  »B
¿
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2Ë8Ä`@Ä`HÄ`b conv2d_16/kernel/Regularizer/Sumhu  »B
Ù
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ä`@Ä`HÄ`Xb9gradient_tape/model/conv2d_10/Conv2D/Conv2DBackpropFilterhuZUÖB
Ù
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ä`@Ä`HÄ`Xb9gradient_tape/model/conv2d_13/Conv2D/Conv2DBackpropFilterhuZUÖB
Q
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2Ä
8ÄX@ÄXHÄXbIsFinite_48hu  »B
Q
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2†8ÄX@ÄXHÄXbIsFinite_34hu  »B
¨
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8ÄX@ÄXHÄXbmodel/conv2d_10/Conv2D/CasthuZUÖB
¨
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8ÄX@ÄXHÄXbmodel/conv2d_13/Conv2D/CasthuZUÖB
≥
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8ÄX@ÄXHÄXb.gradient_tape/model/conv2d_21/Conv2D/Cast/CasthuZUÖB
≥
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8ÄP@ÄPHÄPb.gradient_tape/model/conv2d_10/Conv2D/Cast/CasthuZUÖB
≥
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8ÄP@ÄPHÄPb.gradient_tape/model/conv2d_13/Conv2D/Cast/CasthuZUÖB
¿
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2ê8ÄP@ÄPHÄPb conv2d_11/kernel/Regularizer/Sumhu  »B
Q
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2Ä	8ÄH@ÄHHÄHbIsFinite_18hu  »B
Q
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2Ä	8ÄH@ÄHHÄHbIsFinite_26hu  »B
ß
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8ÄH@ÄHHÄHbAdam/gradients/AddN_15huZUÖB
”
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2{8ÄH@ÄHHÄHb.gradient_tape/conv2d_8/kernel/Regularizer/TilehuZUÖB
â
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2Ù8ÄH@ÄHHÄHbAll_44hu  »B
¿
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2†8ÄH@ÄHHÄHb conv2d_10/kernel/Regularizer/Sumhu  »B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2d8Ä@@Ä@HÄ@b/gradient_tape/conv2d_8/kernel/Regularizer/Mul_1huZUÖB
Î
èvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorReshapingOp<Eigen::IndexList<int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> >, Eigen::TensorReductionOp<Eigen::internal::MaxReducer<float, 0>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorReshapingOp<Eigen::IndexList<int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> >, Eigen::TensorReductionOp<Eigen::internal::MaxReducer<float, 0>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long)@*Ä28Ä@@Ä@HÄ@b:categorical_crossentropy/softmax_cross_entropy_with_logitshuZUÖB
â
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2Ù8Ä@@Ä@HÄ@bAll_36hu  »B
¿
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2†8Ä@@Ä@HÄ@b conv2d_13/kernel/Regularizer/Sumhu  »B
¿
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2¿8Ä@@Ä@HÄ@b conv2d_21/kernel/Regularizer/Sumhu  »B
Ú
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ä@@Ä@HÄ@Xb7gradient_tape/model/conv2d_8/Conv2D/Conv2DBackpropInputhuZUÖB
–
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ä@@Ä@HÄ@Xbmodel/conv2d_8/Conv2DhuZUÖB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2H8Ä8@Ä8HÄ8b/gradient_tape/conv2d_7/kernel/Regularizer/Mul_1huZUÖB
q
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2P8Ä8@Ä8HÄ8b0gradient_tape/conv2d_18/kernel/Regularizer/Mul_1huZUÖB
ß
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä8@Ä8HÄ8bAdam/gradients/AddN_14huZUÖB
ß
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä8@Ä8HÄ8bAdam/gradients/AddN_25huZUÖB
Â
âvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorReshapingOp<Eigen::IndexList<int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> >, Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorReshapingOp<Eigen::IndexList<int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> >, Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long)/*Ä28Ä8@Ä8HÄ8b:categorical_crossentropy/softmax_cross_entropy_with_logitshuZUÖB
â
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2Ë8Ä8@Ä8HÄ8bAll_42hu  »B
â
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2ê8Ä8@Ä8HÄ8bAll_28hu  »B
–
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ä8@Ä8HÄ8Xbmodel/conv2d_7/Conv2DhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2P8Å0@Å0HÅ0b<cond_1/then/_10/cond_1/Adam/Adam/update_32/ResourceApplyAdamhuZUÖB
n
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2d8Ä0@Ä0HÄ0b-gradient_tape/conv2d_8/kernel/Regularizer/MulhuZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2d8Ä0@Ä0HÄ0bmul_14huZUÖB
”
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä28Ä0@Ä0HÄ0b.gradient_tape/conv2d_6/kernel/Regularizer/TilehuZUÖB
‘
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2P8Ä0@Ä0HÄ0b/gradient_tape/conv2d_15/kernel/Regularizer/TilehuZUÖB
‘
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2{8Ä0@Ä0HÄ0b/gradient_tape/conv2d_18/kernel/Regularizer/TilehuZUÖB
”
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2{8Ä0@Ä0HÄ0b.gradient_tape/conv2d_7/kernel/Regularizer/TilehuZUÖB
â
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2Ä8Ä0@Ä0HÄ0bAll_60hu  »B
¶
Àvoid cutlass_cudnn::Kernel<cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4> >(cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4>::Params)6* 28Ä0@Ä0HÄ0Xb8gradient_tape/model/conv2d_5/Conv2D/Conv2DBackpropFilterhu  ñB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2@8Ä0@Ä0HÄ0b<cond_1/then/_10/cond_1/Adam/Adam/update_24/ResourceApplyAdamhuZUÖB
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä2 8Ä0@Ä0HÄ0bAll_32hu  »B
á
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Ä0@Ä0HÄ0b1gradient_tape/model/conv2d_10/BiasAdd/BiasAddGradhuZUÖB
á
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Ä0@Ä0HÄ0b1gradient_tape/model/conv2d_17/BiasAdd/BiasAddGradhuZUÖB
Ú
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ä0@Ä0HÄ0Xb7gradient_tape/model/conv2d_7/Conv2D/Conv2DBackpropInputhuZUÖB
Û
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ä0@Ä0HÄ0Xb8gradient_tape/model/conv2d_7/Conv2D/Conv2DBackpropFilterhuZUÖB
Û
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ä0@Ä0HÄ0Xb8gradient_tape/model/conv2d_8/Conv2D/Conv2DBackpropFilterhuZUÖB
g
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2P8Å(@Å(HÅ(b#conv2d_18/kernel/Regularizer/SquarehuZUÖB
”
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2@8Å(@Å(HÅ(b.gradient_tape/conv2d_9/kernel/Regularizer/TilehuZUÖB
á
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Å(@Å(HÅ(b1gradient_tape/model/conv2d_13/BiasAdd/BiasAddGradhuZUÖB
Q
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2ê8Ä(@Ä(HÄ(bIsFinite_12hu  »B
n
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2H8Ä(@Ä(HÄ(b-gradient_tape/conv2d_7/kernel/Regularizer/MulhuZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2H8Ä(@Ä(HÄ(bmul_12huZUÖB
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2P8Ä(@Ä(HÄ(b.gradient_tape/conv2d_18/kernel/Regularizer/MulhuZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2P8Ä(@Ä(HÄ(bmul_42huZUÖB
f
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2H8Ä(@Ä(HÄ(b"conv2d_7/kernel/Regularizer/SquarehuZUÖB
f
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2d8Ä(@Ä(HÄ(b"conv2d_8/kernel/Regularizer/SquarehuZUÖB
≥
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä(@Ä(HÄ(b.gradient_tape/model/conv2d_18/Conv2D/Cast/CasthuZUÖB
≤
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä(@Ä(HÄ(b-gradient_tape/model/conv2d_8/Conv2D/Cast/CasthuZUÖB
›
ìvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long) *Ä28Ä(@ÄHÄb(ArithmeticOptimizer/AddOpsRewrite_AddN_1huZUÖB
”
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä28Ä(@Ä(HÄ(b.gradient_tape/conv2d_5/kernel/Regularizer/TilehuZUÖB
â
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2ê8Ä(@Ä(HÄ(bAll_18hu  »B
â
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2†8Ä(@Ä(HÄ(bAll_48hu  »B
â
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2¥8Ä(@Ä(HÄ(bAll_34hu  »B
â
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2»8Ä(@Ä(HÄ(bAll_20hu  »B
¶
Àvoid cutlass_cudnn::Kernel<cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4> >(cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4>::Params)6* 28Ä(@Ä(HÄ(Xb8gradient_tape/model/conv2d_4/Conv2D/Conv2DBackpropFilterhu  ñB
Ä
§void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*) *Ä28Ä(@Ä(HÄ(Xb8gradient_tape/model/conv2d_3/Conv2D/Conv2DBackpropFilterhu  »B
Ç
§void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*) *Ä2Ä8Ä(@Ä(HÄ(Xb9gradient_tape/model/conv2d_18/Conv2D/Conv2DBackpropFilterhu  »B
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä(@Ä(HÄ(b<cond_1/then/_10/cond_1/Adam/Adam/update_45/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä(@Ä(HÄ(b<cond_1/then/_10/cond_1/Adam/Adam/update_63/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2@8Ä(@Ä(HÄ(b<cond_1/then/_10/cond_1/Adam/Adam/update_16/ResourceApplyAdamhuZUÖB
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä(@Ä(HÄ(bAll_54hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä(@Ä(HÄ(bAll_55hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä2 8Ä(@Ä(HÄ(bAll_16hu  »B
á
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Ä(@Ä(HÄ(b1gradient_tape/model/conv2d_14/BiasAdd/BiasAddGradhuZUÖB
á
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Ä(@Ä(HÄ(b1gradient_tape/model/conv2d_16/BiasAdd/BiasAddGradhuZUÖB
á
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Ä(@Ä(HÄ(b1gradient_tape/model/conv2d_20/BiasAdd/BiasAddGradhuZUÖB
á
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Ä(@Ä(HÄ(b1gradient_tape/model/conv2d_23/BiasAdd/BiasAddGradhuZUÖB
Ü
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Ä(@Ä(HÄ(b0gradient_tape/model/conv2d_7/BiasAdd/BiasAddGradhuZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Å @Å HÅ bmul_34huZUÖB
ø
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2P8Å @Å HÅ b conv2d_18/kernel/Regularizer/Sumhu  »B
ê
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Å @Å HÅ b9cond_1/then/_10/cond_1/Adam/Adam/update/ResourceApplyAdamhuZUÖB
á
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Å @Å HÅ b1gradient_tape/model/conv2d_12/BiasAdd/BiasAddGradhuZUÖB
˝
ßvoid tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>)*Ä2@8Å @Å HÅ b4model/dropout_4/dropout/random_uniform/RandomUniformhuZUÖB
Q
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2†8Ä @Ä HÄ bIsFinite_10hu  »B
Q
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2¿8Ä @Ä HÄ bIsFinite_40hu  »B
n
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä @Ä HÄ b-gradient_tape/conv2d_9/kernel/Regularizer/MulhuZUÖB
q
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä @Ä HÄ b0gradient_tape/conv2d_15/kernel/Regularizer/Mul_1huZUÖB
i
 ampere_fp16_sgemm_fp16_128x32_tn9ÄÄ*Ä28Ä @Ä HÄ Xb"gradient_tape/model/dense_2/MatMulhuZUÖB
¨
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä @Ä HÄ bmodel/conv2d_18/Conv2D/CasthuZUÖB
´
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä @Ä HÄ bmodel/conv2d_8/Conv2D/CasthuZUÖB
ß
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä @Ä HÄ bAdam/gradients/AddN_13huZUÖB
ß
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä @Ä HÄ bAdam/gradients/AddN_16huZUÖB
ß
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2P8Ä @Ä HÄ bAdam/gradients/AddN_22huZUÖB
æ	
Ávoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_squared_difference_op<float>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_squared_difference_op<float>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, int) *Ä2@8Ä @Ä HÄ b5model/batch_normalization_8/moments/SquaredDifferencehuZUÖB
ˇ	
£	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)&*Ä28Ä @Ä HÄ b:categorical_crossentropy/softmax_cross_entropy_with_logitshuZUÖB
≈
Èvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_quotient_op<float, float>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_quotient_op<float, float>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long).*Ä28Ä @Ä HÄ b:categorical_crossentropy/softmax_cross_entropy_with_logitshuZUÖB
”
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä28Ä @Ä HÄ b.gradient_tape/conv2d_4/kernel/Regularizer/TilehuZUÖB
≠
—void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorEvalToOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorForcedEvalOp<Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_log_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorEvalToOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorForcedEvalOp<Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_log_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>, long)(*Ä28Ä @Ä HÄ b:categorical_crossentropy/softmax_cross_entropy_with_logitshuZUÖB
à
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä228Ä @Ä HÄ bAll_12hu  »B
â
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2ê8Ä @Ä HÄ bAll_26hu  »B
æ
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2H8Ä @Ä HÄ bconv2d_7/kernel/Regularizer/Sumhu  »B
æ
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2d8Ä @Ä HÄ bconv2d_8/kernel/Regularizer/Sumhu  »B
Ö
€void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *Ä28Ä @Ä HÄ bAll_56hu¶™¶B
π
˘void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float>, float>(float*, float*, int, tensorflow::functor::Sum<float>, float)0*Ä28Ä @Ä HÄ bdense/kernel/Regularizer/Sumhu  »B
‡
§void cutlass::Kernel<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2>(cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2::Params)_ Ä§*Ä28Ä @Ä HÄ Xbmodel/dense_2/MatMulhugUÖA
Ó
§void cutlass::Kernel<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nt_align2>(cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nt_align2::Params)_ Ä¿*Ä2@8Ä @Ä HÄ b$gradient_tape/model/dense_2/MatMul_1hugUÖA
Å
§void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*) *Ä2Ä8Ä @Ä HÄ Xb8gradient_tape/model/conv2d_9/Conv2D/Conv2DBackpropFilterhu  »B
€
§void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*) *Ä2Ä8Ä @Ä HÄ Xbmodel/dense/MatMulhu  »B
ˇ
¢void splitKreduce_kernel<float, __half, float, __half>(cublasSplitKParams<float>, float const*, __half const*, __half*, float const*, float const*, __half const*) *Ä2Ä8Ä @Ä HÄ Xb8gradient_tape/model/conv2d_6/Conv2D/Conv2DBackpropFilterhu  »B
Ê
©void tensorflow::(anonymous namespace)::GenerateNormalizedProb<Eigen::half, float, 8>(Eigen::half const*, float const*, Eigen::half const*, Eigen::half*, int, int, bool)*Ä28Ä @Ä HÄ bmodel/activation_10/Softmaxhu  »B
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_11/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_13/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_14/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_15/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_17/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_19/ResourceApplyAdamhuZUÖB
í
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b;cond_1/then/_10/cond_1/Adam/Adam/update_2/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_21/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_27/ResourceApplyAdamhuZUÖB
í
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b;cond_1/then/_10/cond_1/Adam/Adam/update_3/ResourceApplyAdamhuZUÖB
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
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_43/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_49/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_53/ResourceApplyAdamhuZUÖB
í
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b;cond_1/then/_10/cond_1/Adam/Adam/update_7/ResourceApplyAdamhuZUÖB
í
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b;cond_1/then/_10/cond_1/Adam/Adam/update_9/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_47/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_57/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_58/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_59/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_61/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_62/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_54/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_55/ResourceApplyAdamhuZUÖB
í
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b;cond_1/then/_10/cond_1/Adam/Adam/update_4/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_64/ResourceApplyAdamhuZUÖB
í
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b;cond_1/then/_10/cond_1/Adam/Adam/update_8/ResourceApplyAdamhuZUÖB
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä @Ä HÄ bAll_41hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä @Ä HÄ bAll_46hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä @Ä HÄ bAll_49hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä @Ä HÄ bAll_57hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä @Ä HÄ bAll_59hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä @Ä HÄ bAll_61hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä @Ä HÄ bAll_62hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä2 8Ä @Ä HÄ bAll_24hu  »B
Ö
¬void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*Ä2 8Ä @Ä HÄ bconv2d_9/kernel/Regularizer/Sumhu  »B
á
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Ä @Ä HÄ b1gradient_tape/model/conv2d_11/BiasAdd/BiasAddGradhuZUÖB
Ü
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Ä @Ä HÄ b0gradient_tape/model/conv2d_8/BiasAdd/BiasAddGradhuZUÖB
Ü
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Ä @Ä HÄ b0gradient_tape/model/conv2d_9/BiasAdd/BiasAddGradhuZUÖB
˝
ßvoid tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>)*Ä2@8Ä @Ä HÄ b4model/dropout_5/dropout/random_uniform/RandomUniformhuZUÖB
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Å@ÅHÅbIsFinite_57hu  »B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Å@ÅHÅbmul_17huZUÖB
l
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2@8Å@ÅHÅb+model/batch_normalization_9/batchnorm/mul_1huZUÖB
ê
Ÿvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Å@ÅHÅbAssignAddVariableOp_4huZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Å@ÅHÅb<cond_1/then/_10/cond_1/Adam/Adam/update_25/ResourceApplyAdamhuZUÖB
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä2 8Å@ÅHÅbAll_64hu  »B
ï
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2@8Å@ÅHÅb-gradient_tape/model/dense/BiasAdd/BiasAddGradhuZUÖB
l
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb)model/batch_normalization_8/batchnorm/addhuZUÖB
l
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb)model/batch_normalization_9/batchnorm/addhuZUÖB
n
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2@8Ä@ÄHÄb+model/batch_normalization_8/batchnorm/add_1huZUÖB
n
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2@8Ä@ÄHÄb+model/batch_normalization_9/batchnorm/add_1huZUÖB
b
"AddV2_GPU_DT_INT64_DT_INT64_kernel*Ä28Ä@ÄHÄbcond_1/then/_10/cond_1/Adam/addhuZUÖB
z
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2@8Ä@ÄHÄb9gradient_tape/model/batch_normalization_8/moments/truedivhuZUÖB
|
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2@8Ä@ÄHÄb;gradient_tape/model/batch_normalization_9/moments/truediv_1huZUÖB
l
'GreaterEqual_GPU_DT_HALF_DT_BOOL_kernel*Ä28Ä@ÄHÄb$model/dropout_4/dropout/GreaterEqualhuZUÖB
g
(GreaterEqual_GPU_DT_INT64_DT_BOOL_kernel*Ä28Ä@ÄHÄbcond/then/_0/cond/GreaterEqualhuZUÖB
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_11hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_13hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_14hu  »B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄb
IsFinite_2hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_23hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_27hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_35hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_43hu  »B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄb
IsFinite_5hu  »B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄb
IsFinite_6hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_55hu  »B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄb
IsFinite_8hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2@8Ä@ÄHÄbIsFinite_16hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2P8Ä@ÄHÄbIsFinite_32hu  »B
D
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbMulhuZUÖB
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb conv2d_10/kernel/Regularizer/mulhuZUÖB
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb conv2d_11/kernel/Regularizer/mulhuZUÖB
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb conv2d_16/kernel/Regularizer/mulhuZUÖB
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb conv2d_18/kernel/Regularizer/mulhuZUÖB
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb conv2d_22/kernel/Regularizer/mulhuZUÖB
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbconv2d_6/kernel/Regularizer/mulhuZUÖB
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbconv2d_8/kernel/Regularizer/mulhuZUÖB
_
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbdense_1/kernel/Regularizer/mulhuZUÖB
n
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb-gradient_tape/conv2d_3/kernel/Regularizer/MulhuZUÖB
~
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb=gradient_tape/model/batch_normalization_8/batchnorm/mul/Mul_1huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_15huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_23huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_24huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_27huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_29huZUÖB
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_3huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_32huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_33huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_35huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_37huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_41huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_45huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_47huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_49huZUÖB
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_5huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_51huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_56huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_59huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_63huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_67huZUÖB
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_7huZUÖB
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_8huZUÖB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb/gradient_tape/conv2d_4/kernel/Regularizer/Mul_1huZUÖB
Ä
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb?gradient_tape/model/batch_normalization_8/batchnorm/mul_2/Mul_1huZUÖB
|
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb;gradient_tape/model/batch_normalization_9/batchnorm/mul/MulhuZUÖB
~
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb=gradient_tape/model/batch_normalization_9/batchnorm/mul_2/MulhuZUÖB
j
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb)model/batch_normalization_8/batchnorm/mulhuZUÖB
l
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb+model/batch_normalization_8/batchnorm/mul_2huZUÖB
j
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb)model/batch_normalization_9/batchnorm/mulhuZUÖB
l
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb+model/batch_normalization_9/batchnorm/mul_2huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_60huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_64huZUÖB
n
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb-gradient_tape/conv2d_5/kernel/Regularizer/MulhuZUÖB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb/gradient_tape/conv2d_5/kernel/Regularizer/Mul_1huZUÖB
m
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb,gradient_tape/dense_2/kernel/Regularizer/MulhuZUÖB
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb.gradient_tape/dense_2/kernel/Regularizer/Mul_1huZUÖB
n
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb-gradient_tape/conv2d_6/kernel/Regularizer/MulhuZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_10huZUÖB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb/gradient_tape/conv2d_6/kernel/Regularizer/Mul_1huZUÖB
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb.gradient_tape/conv2d_12/kernel/Regularizer/MulhuZUÖB
v
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb5gradient_tape/model/batch_normalization_8/moments/MulhuZUÖB
v
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb5gradient_tape/model/batch_normalization_9/moments/MulhuZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_18huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_26huZUÖB
q
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb0gradient_tape/conv2d_12/kernel/Regularizer/Mul_1huZUÖB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb/gradient_tape/conv2d_9/kernel/Regularizer/Mul_1huZUÖB
Ä
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb?gradient_tape/model/batch_normalization_8/batchnorm/mul_1/Mul_1huZUÖB
x
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb7gradient_tape/model/batch_normalization_8/moments/mul_1huZUÖB
Ä
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb?gradient_tape/model/batch_normalization_9/batchnorm/mul_1/Mul_1huZUÖB
x
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb7gradient_tape/model/batch_normalization_9/moments/mul_1huZUÖB
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb.gradient_tape/conv2d_15/kernel/Regularizer/MulhuZUÖB
~
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2@8Ä@ÄHÄb=gradient_tape/model/batch_normalization_8/batchnorm/mul_1/MulhuZUÖB
~
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2@8Ä@ÄHÄb=gradient_tape/model/batch_normalization_9/batchnorm/mul_1/MulhuZUÖB
l
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2@8Ä@ÄHÄb+model/batch_normalization_8/batchnorm/mul_1huZUÖB
h
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä28Ä@ÄHÄb)gradient_tape/model/dropout_4/dropout/MulhuZUÖB
j
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä28Ä@ÄHÄb+gradient_tape/model/dropout_4/dropout/Mul_1huZUÖB
j
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä28Ä@ÄHÄb+gradient_tape/model/dropout_5/dropout/Mul_1huZUÖB
Z
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä28Ä@ÄHÄbmodel/dropout_4/dropout/MulhuZUÖB
Z
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä28Ä@ÄHÄbmodel/dropout_5/dropout/MulhuZUÖB
\
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä28Ä@ÄHÄbmodel/dropout_5/dropout/Mul_1huZUÖB
`
 Pow_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbcond_1/then/_10/cond_1/Adam/PowhuZUÖB
b
 Pow_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb!cond_1/then/_10/cond_1/Adam/Pow_1huZUÖB
f
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb"conv2d_3/kernel/Regularizer/SquarehuZUÖB
f
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb"conv2d_5/kernel/Regularizer/SquarehuZUÖB
e
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb!dense_2/kernel/Regularizer/SquarehuZUÖB
f
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb"conv2d_6/kernel/Regularizer/SquarehuZUÖB
g
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb#conv2d_12/kernel/Regularizer/SquarehuZUÖB
g
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb#conv2d_15/kernel/Regularizer/SquarehuZUÖB
p
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb/model/batch_normalization_8/AssignMovingAvg/subhuZUÖB
r
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb1model/batch_normalization_8/AssignMovingAvg_1/subhuZUÖB
v
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2@8Ä@ÄHÄb5gradient_tape/model/batch_normalization_8/moments/subhuZUÖB
≤
htensorflow::functor::ReluGradHalfKernelVector(Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä28Ä@ÄHÄb)gradient_tape/model/activation_8/ReluGradhu  »B
´
Ìvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄbmodel/dropout_5/dropout/CasthuZUÖB
ï
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbCasthuZUÖB
é
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb}categorical_crossentropy/softmax_cross_entropy_with_logits/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Cast_2huZUÖB
≠
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel/conv2d_12/BiasAdd/CasthuZUÖB
≠
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel/conv2d_14/BiasAdd/CasthuZUÖB
≠
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel/conv2d_16/BiasAdd/CasthuZUÖB
≠
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel/conv2d_19/BiasAdd/CasthuZUÖB
≠
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel/conv2d_20/BiasAdd/CasthuZUÖB
¨
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel/conv2d_4/BiasAdd/CasthuZUÖB
¨
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel/conv2d_5/BiasAdd/CasthuZUÖB
´
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel/dense_2/BiasAdd/CasthuZUÖB
´
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel/dense_1/BiasAdd/CasthuZUÖB
ƒ
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄb3gradient_tape/model/batch_normalization_9/Cast/CasthuZUÖB
≥
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄb"model/batch_normalization_9/Cast_1huZUÖB
¨
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄbmodel/conv2d_12/Conv2D/CasthuZUÖB
¨
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2P8Ä@ÄHÄbmodel/conv2d_15/Conv2D/CasthuZUÖB
´
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä@ÄHÄbmodel/conv2d_7/Conv2D/CasthuZUÖB
†	
Ávoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄbmodel/activation_8/ReluhuZUÖB
†	
Ávoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄbmodel/activation_9/ReluhuZUÖB
‡
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä28Ä@ÄHÄb;gradient_tape/categorical_crossentropy/weighted_loss/Tile_1huZUÖB
∫
€void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_rsqrt_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_rsqrt_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb=gradient_tape/model/batch_normalization_8/batchnorm/RsqrtGradhuZUÖB
∫
€void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_rsqrt_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_rsqrt_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb=gradient_tape/model/batch_normalization_9/batchnorm/RsqrtGradhuZUÖB
Ç
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb}categorical_crossentropy/softmax_cross_entropy_with_logits/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Cast_1huZUÖB
ƒ
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb?categorical_crossentropy/softmax_cross_entropy_with_logits/CasthuZUÖB
¥
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb/gradient_tape/model/conv2d_10/BiasAdd/Cast/CasthuZUÖB
¥
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb/gradient_tape/model/conv2d_11/BiasAdd/Cast/CasthuZUÖB
¥
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb/gradient_tape/model/conv2d_14/BiasAdd/Cast/CasthuZUÖB
¥
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb/gradient_tape/model/conv2d_21/BiasAdd/Cast/CasthuZUÖB
≤
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb-gradient_tape/model/conv2d_4/Conv2D/Cast/CasthuZUÖB
≥
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb.gradient_tape/model/conv2d_8/BiasAdd/Cast/CasthuZUÖB
≤
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb-gradient_tape/model/dense_1/BiasAdd/Cast/CasthuZUÖB
≤
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb-gradient_tape/model/conv2d_5/Conv2D/Cast/CasthuZUÖB
≤
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb-gradient_tape/model/conv2d_6/Conv2D/Cast/CasthuZUÖB
∫
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄb5gradient_tape/model/batch_normalization_8/Cast_1/CasthuZUÖB
∫
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄb5gradient_tape/model/batch_normalization_9/Cast_1/CasthuZUÖB
≥
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄb.gradient_tape/model/conv2d_12/Conv2D/Cast/CasthuZUÖB
≤
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄb-gradient_tape/model/conv2d_9/Conv2D/Cast/CasthuZUÖB
•
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄb model/batch_normalization_9/CasthuZUÖB
≥
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2P8Ä@ÄHÄb.gradient_tape/model/conv2d_15/Conv2D/Cast/CasthuZUÖB
≤
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä@ÄHÄb-gradient_tape/model/conv2d_7/Conv2D/Cast/CasthuZUÖB
˝
’void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbCast_7huZUÖB
˚
”void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbCast_8huZUÖB
≠
”void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb8gradient_tape/model/batch_normalization_8/moments/Cast_1huZUÖB
æ
Òvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb+model/batch_normalization_8/AssignMovingAvghuZUÖB
æ
Òvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb+model/batch_normalization_9/AssignMovingAvghuZUÖB
§
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbAdam/gradients/AddNhuZUÖB
ß
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbAdam/gradients/AddN_33huZUÖB
ß
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄbAdam/gradients/AddN_19huZUÖB
ö
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbAssignAddVariableOp_1huZUÖB
ˆ	
ø	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄbAdam/gradients/AddN_1huZUÖB
ˆ	
ø	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄbAdam/gradients/AddN_3huZUÖB
“
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä28Ä@ÄHÄb-gradient_tape/dense_2/kernel/Regularizer/TilehuZUÖB
‚
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2@8Ä@ÄHÄb=gradient_tape/model/batch_normalization_8/moments/BroadcastTohuZUÖB
‰
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2@8Ä@ÄHÄb?gradient_tape/model/batch_normalization_8/moments/BroadcastTo_1huZUÖB
‚
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2@8Ä@ÄHÄb=gradient_tape/model/batch_normalization_9/moments/BroadcastTohuZUÖB
‰
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2@8Ä@ÄHÄb?gradient_tape/model/batch_normalization_9/moments/BroadcastTo_1huZUÖB
æ	
Ávoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_squared_difference_op<float>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_squared_difference_op<float>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, int) *Ä2@8Ä@ÄHÄb5model/batch_normalization_9/moments/SquaredDifferencehuZUÖB
”
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä28Ä@ÄHÄb.gradient_tape/conv2d_3/kernel/Regularizer/TilehuZUÖB
π
Ÿvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb>cond/then/_0/cond/cond/else/_843/cond/cond/AssignAddVariableOphuZUÖB
≥
ãvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorTupleReducerOp<Eigen::internal::ArgMaxTupleReducer<Eigen::Tuple<long, float> >, Eigen::array<long, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorTupleReducerOp<Eigen::internal::ArgMaxTupleReducer<Eigen::Tuple<long, float> >, Eigen::array<long, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long) *Ä28Ä@ÄHÄbArgMaxhuZUÖB
µ
ãvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorTupleReducerOp<Eigen::internal::ArgMaxTupleReducer<Eigen::Tuple<long, float> >, Eigen::array<long, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorTupleReducerOp<Eigen::internal::ArgMaxTupleReducer<Eigen::Tuple<long, float> >, Eigen::array<long, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long) *Ä28Ä@ÄHÄbArgMax_1huZUÖB
à
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2$8Ä@ÄHÄbAll_10hu  »B
à
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2(8Ä@ÄHÄbAll_40hu  »B
Ö
€void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *Ä28Ä@ÄHÄbAll_12hu¶™¶B
Ö
€void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *Ä28Ä@ÄHÄbAll_18hu¶™¶B
Ö
€void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *Ä28Ä@ÄHÄbAll_28hu¶™¶B
Ö
€void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *Ä28Ä@ÄHÄbAll_36hu¶™¶B
Ö
€void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *Ä28Ä@ÄHÄbAll_40hu¶™¶B
Ö
€void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *Ä28Ä@ÄHÄbAll_42hu¶™¶B
Ö
€void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *Ä28Ä@ÄHÄbAll_48hu¶™¶B
Ö
€void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *Ä28Ä@ÄHÄbAll_50hu¶™¶B
Ö
€void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *Ä28Ä@ÄHÄbAll_52hu¶™¶B
Ö
€void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *Ä28Ä@ÄHÄbAll_60hu¶™¶B
Ω
˘void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float>, float>(float*, float*, int, tensorflow::functor::Sum<float>, float)0*Ä28Ä@ÄHÄb conv2d_16/kernel/Regularizer/Sumhu  »B
Ω
˘void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float>, float>(float*, float*, int, tensorflow::functor::Sum<float>, float)0*Ä28Ä@ÄHÄb conv2d_17/kernel/Regularizer/Sumhu  »B
Ω
˘void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float>, float>(float*, float*, int, tensorflow::functor::Sum<float>, float)0*Ä28Ä@ÄHÄb conv2d_19/kernel/Regularizer/Sumhu  »B
Ω
˘void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float>, float>(float*, float*, int, tensorflow::functor::Sum<float>, float)0*Ä28Ä@ÄHÄb conv2d_21/kernel/Regularizer/Sumhu  »B
Ω
˘void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float>, float>(float*, float*, int, tensorflow::functor::Sum<float>, float)0*Ä28Ä@ÄHÄb conv2d_22/kernel/Regularizer/Sumhu  »B
Ω
˘void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float>, float>(float*, float*, int, tensorflow::functor::Sum<float>, float)0*Ä28Ä@ÄHÄb conv2d_23/kernel/Regularizer/Sumhu  »B
ª
˘void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float>, float>(float*, float*, int, tensorflow::functor::Sum<float>, float)0*Ä28Ä@ÄHÄbdense_1/kernel/Regularizer/Sumhu  »B
‹
§void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*) *Ä28Ä@ÄHÄXbmodel/dense_2/MatMulhu  »B
Ç
§void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*) *Ä2Ä8Ä@ÄHÄXb9gradient_tape/model/conv2d_12/Conv2D/Conv2DBackpropFilterhu  »B
Ç
§void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*) *Ä2Ä8Ä@ÄHÄXb9gradient_tape/model/conv2d_15/Conv2D/Conv2DBackpropFilterhu  »B
ª
évoid tensorflow::(anonymous namespace)::concat_fixed_kernel<bool, int>(tensorflow::GpuDeviceArrayStruct<bool const*, 8>, int, int, int, bool*)*B28Ä@ÄHÄbAll_66/inputhu∞öB
†
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2@8Ä@ÄHÄbmodel/dense/BiasAddhuZUÖB
¢
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2@8Ä@ÄHÄbmodel/dense_1/BiasAddhuZUÖB
í
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb;cond_1/then/_10/cond_1/Adam/Adam/update_1/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb<cond_1/then/_10/cond_1/Adam/Adam/update_22/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb<cond_1/then/_10/cond_1/Adam/Adam/update_23/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb<cond_1/then/_10/cond_1/Adam/Adam/update_29/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb<cond_1/then/_10/cond_1/Adam/Adam/update_30/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb<cond_1/then/_10/cond_1/Adam/Adam/update_38/ResourceApplyAdamhuZUÖB
í
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb;cond_1/then/_10/cond_1/Adam/Adam/update_5/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb<cond_1/then/_10/cond_1/Adam/Adam/update_51/ResourceApplyAdamhuZUÖB
í
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb;cond_1/then/_10/cond_1/Adam/Adam/update_6/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb<cond_1/then/_10/cond_1/Adam/Adam/update_65/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb<cond_1/then/_10/cond_1/Adam/Adam/update_46/ResourceApplyAdamhuZUÖB
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_13hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_14hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_19hu  »B
◊
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_2hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_21hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_22hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_27hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_29hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_30hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_31hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_35hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_38hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_39hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_43hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_45hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_47hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_51hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_53hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_58hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_63hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_65hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_66hu  »B
◊
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_7hu  »B
◊
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_4hu  »B
◊
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä2 8Ä@ÄHÄbAll_8hu  »B
Î
¬void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*Ä28Ä@ÄHÄbSum_2hu  »B
Ö
¬void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*Ä28Ä@ÄHÄbconv2d_4/kernel/Regularizer/Sumhu  »B
Ö
¬void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*Ä28Ä@ÄHÄbconv2d_5/kernel/Regularizer/Sumhu  »B
Ü
¬void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*Ä2 8Ä@ÄHÄb conv2d_12/kernel/Regularizer/Sumhu  »B
Ü
¬void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*Ä2 8Ä@ÄHÄb conv2d_15/kernel/Regularizer/Sumhu  »B
Ö
¬void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*Ä2 8Ä@ÄHÄbconv2d_6/kernel/Regularizer/Sumhu  »B
Ö
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2 8Ä@ÄHÄb0gradient_tape/model/conv2d_3/BiasAdd/BiasAddGradhuZUÖB
Ö
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2 8Ä@ÄHÄb0gradient_tape/model/conv2d_4/BiasAdd/BiasAddGradhuZUÖB
Ö
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2@8Ä@ÄHÄb0gradient_tape/model/conv2d_5/BiasAdd/BiasAddGradhuZUÖB
á
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Ä@ÄHÄb1gradient_tape/model/conv2d_15/BiasAdd/BiasAddGradhuZUÖB
Ü
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Ä@ÄHÄb0gradient_tape/model/conv2d_6/BiasAdd/BiasAddGradhuZUÖB
Ñ
≈void tensorflow::functor::CleanupSegments<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)* 28Ä@ÄHÄbdense_2/kernel/Regularizer/SumhuMUB
ó
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2@8Ä@ÄHÄb/gradient_tape/model/dense_1/BiasAdd/BiasAddGradhuZUÖB
¶
√void tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)Ä!*  2@8Ä@ÄHÄb?gradient_tape/model/batch_normalization_8/batchnorm/add_1/Sum_1huZUÖB
¶
√void tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)Ä!*  2@8Ä@ÄHÄb?gradient_tape/model/batch_normalization_8/batchnorm/mul_1/Sum_1huZUÖB
¶
√void tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)Ä!*  2@8Ä@ÄHÄb?gradient_tape/model/batch_normalization_9/batchnorm/add_1/Sum_1huZUÖB
¶
√void tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)Ä!*  2@8Ä@ÄHÄb?gradient_tape/model/batch_normalization_9/batchnorm/mul_1/Sum_1huZUÖB
Õ
Åvoid tensorflow::functor::ColumnReduceKernel<float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, tensorflow::functor::Sum<float> >(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)Ä!*  2@8Ä@ÄHÄb(model/batch_normalization_8/moments/meanhuZUÖB
—
Åvoid tensorflow::functor::ColumnReduceKernel<float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, tensorflow::functor::Sum<float> >(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)Ä!*  2@8Ä@ÄHÄb,model/batch_normalization_8/moments/variancehuZUÖB
Õ
Åvoid tensorflow::functor::ColumnReduceKernel<float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, tensorflow::functor::Sum<float> >(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)Ä!*  2@8Ä@ÄHÄb(model/batch_normalization_9/moments/meanhuZUÖB
—
Åvoid tensorflow::functor::ColumnReduceKernel<float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, tensorflow::functor::Sum<float> >(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)Ä!*  2@8Ä@ÄHÄb,model/batch_normalization_9/moments/variancehuZUÖB
£
–void tensorflow::functor::ColumnReduceMax16ColumnsKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿* 28Ä@ÄHÄb/gradient_tape/model/dense_2/BiasAdd/BiasAddGradhu  ØB
˛
¡void tensorflow::functor::RowReduceKernel<Eigen::half const*, Eigen::half*, cub::Max>(Eigen::half const*, Eigen::half*, int, int, cub::Max, std::iterator_traits<Eigen::half const*>::value_type)*Ä28Ä@ÄHÄbmodel/activation_10/Softmaxhu  »B
–
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä28Ä@ÄHÄXbmodel/conv2d_4/Conv2DhuZUÖB
–
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä28Ä@ÄHÄXbmodel/conv2d_5/Conv2DhuZUÖB
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Å@ÅHÅbconv2d_3/kernel/Regularizer/mulhuZUÖB
ß
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Å@ÅHÅbAdam/gradients/AddN_12huZUÖB
X
"AddV2_GPU_DT_INT64_DT_INT64_kernel*Ä28Ä@ÄHÄbcond/then/_0/cond/addhuZUÖB
|
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2@8Ä@ÄHÄb;gradient_tape/model/batch_normalization_8/moments/truediv_1huZUÖB
z
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2@8Ä@ÄHÄb9gradient_tape/model/batch_normalization_9/moments/truedivhuZUÖB
G
!Equal_GPU_DT_INT64_DT_BOOL_kernel*Ä28Ä@ÄHÄbEqualhuZUÖB
l
'GreaterEqual_GPU_DT_HALF_DT_BOOL_kernel*Ä28Ä@ÄHÄb$model/dropout_5/dropout/GreaterEqualhuZUÖB
M
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinitehu  »B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄb
IsFinite_1hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_15hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_17hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_19hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_21hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_22hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_25hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_29hu  »B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄb
IsFinite_3hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_30hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_31hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_33hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_37hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_38hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_39hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_41hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_45hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_49hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_51hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_53hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_65hu  »B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄb
IsFinite_7hu  »B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄb
IsFinite_9hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_46hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_47hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_58hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_59hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_61hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_62hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_63hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_54hu  »B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄb
IsFinite_4hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_64hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2@8Ä@ÄHÄbIsFinite_24hu  »B
P
%LogicalAnd_GPU_DT_BOOL_DT_BOOL_kernel*Ä28Ä@ÄHÄb
LogicalAndhuZUÖB
ç
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbLgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mulhuZUÖB
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb conv2d_12/kernel/Regularizer/mulhuZUÖB
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb conv2d_13/kernel/Regularizer/mulhuZUÖB
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb conv2d_14/kernel/Regularizer/mulhuZUÖB
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb conv2d_15/kernel/Regularizer/mulhuZUÖB
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb conv2d_17/kernel/Regularizer/mulhuZUÖB
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb conv2d_19/kernel/Regularizer/mulhuZUÖB
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb conv2d_20/kernel/Regularizer/mulhuZUÖB
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb conv2d_21/kernel/Regularizer/mulhuZUÖB
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb conv2d_23/kernel/Regularizer/mulhuZUÖB
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbconv2d_4/kernel/Regularizer/mulhuZUÖB
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbconv2d_5/kernel/Regularizer/mulhuZUÖB
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbconv2d_7/kernel/Regularizer/mulhuZUÖB
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbconv2d_9/kernel/Regularizer/mulhuZUÖB
]
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbdense/kernel/Regularizer/mulhuZUÖB
_
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbdense_2/kernel/Regularizer/mulhuZUÖB
t
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb3gradient_tape/conv2d_3/kernel/Regularizer/mul/Mul_1huZUÖB
n
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb-gradient_tape/conv2d_4/kernel/Regularizer/MulhuZUÖB
~
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb=gradient_tape/model/batch_normalization_9/batchnorm/mul/Mul_1huZUÖB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb/model/batch_normalization_8/AssignMovingAvg/mulhuZUÖB
r
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb1model/batch_normalization_8/AssignMovingAvg_1/mulhuZUÖB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb/model/batch_normalization_9/AssignMovingAvg/mulhuZUÖB
r
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb1model/batch_normalization_9/AssignMovingAvg_1/mulhuZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_11huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_13huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_16huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_19huZUÖB
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_2huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_21huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_25huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_31huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_39huZUÖB
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_4huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_40huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_43huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_48huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_53huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_55huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_57huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_61huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_65huZUÖB
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_9huZUÖB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb/gradient_tape/conv2d_3/kernel/Regularizer/Mul_1huZUÖB
|
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb;gradient_tape/model/batch_normalization_8/batchnorm/mul/MulhuZUÖB
~
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb=gradient_tape/model/batch_normalization_8/batchnorm/mul_2/MulhuZUÖB
Ä
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb?gradient_tape/model/batch_normalization_9/batchnorm/mul_2/Mul_1huZUÖB
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_6huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_66huZUÖB
h
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä28Ä@ÄHÄb)gradient_tape/model/dropout_5/dropout/MulhuZUÖB
\
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä28Ä@ÄHÄbmodel/dropout_4/dropout/Mul_1huZUÖB
|
 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb;gradient_tape/model/batch_normalization_8/batchnorm/sub/Neghu  »B
|
 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb;gradient_tape/model/batch_normalization_9/batchnorm/sub/Neghu  »B
n
"Rsqrt_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb+model/batch_normalization_8/batchnorm/Rsqrthu  »B
n
"Rsqrt_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb+model/batch_normalization_9/batchnorm/Rsqrthu  »B
f
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb"conv2d_4/kernel/Regularizer/SquarehuZUÖB
f
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb"conv2d_9/kernel/Regularizer/SquarehuZUÖB
j
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb)model/batch_normalization_8/batchnorm/subhuZUÖB
p
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb/model/batch_normalization_9/AssignMovingAvg/subhuZUÖB
r
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb1model/batch_normalization_9/AssignMovingAvg_1/subhuZUÖB
j
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb)model/batch_normalization_9/batchnorm/subhuZUÖB
v
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2@8Ä@ÄHÄb5gradient_tape/model/batch_normalization_9/moments/subhuZUÖB
≤
htensorflow::functor::ReluGradHalfKernelVector(Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä28Ä@ÄHÄb)gradient_tape/model/activation_9/ReluGradhu  »B
´
Ìvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄbmodel/dropout_4/dropout/CasthuZUÖB
æ
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb-categorical_crossentropy/weighted_loss/Cast_1huZUÖB
Ê
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbUgradient_tape/Cast_3/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_CasthuZUÖB
†
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbégradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/Cast/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_CasthuZUÖB
ã
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbzgradient_tape/categorical_crossentropy/weighted_loss/Cast/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_CasthuZUÖB
≠
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel/conv2d_10/BiasAdd/CasthuZUÖB
≠
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel/conv2d_11/BiasAdd/CasthuZUÖB
≠
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel/conv2d_13/BiasAdd/CasthuZUÖB
≠
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel/conv2d_15/BiasAdd/CasthuZUÖB
≠
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel/conv2d_17/BiasAdd/CasthuZUÖB
≠
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel/conv2d_18/BiasAdd/CasthuZUÖB
≠
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel/conv2d_21/BiasAdd/CasthuZUÖB
≠
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel/conv2d_22/BiasAdd/CasthuZUÖB
≠
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel/conv2d_23/BiasAdd/CasthuZUÖB
¨
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel/conv2d_3/BiasAdd/CasthuZUÖB
´
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel/conv2d_3/Conv2D/CasthuZUÖB
´
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel/conv2d_4/Conv2D/CasthuZUÖB
¨
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel/conv2d_6/BiasAdd/CasthuZUÖB
¨
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel/conv2d_7/BiasAdd/CasthuZUÖB
¨
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel/conv2d_8/BiasAdd/CasthuZUÖB
¨
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel/conv2d_9/BiasAdd/CasthuZUÖB
©
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel/dense/BiasAdd/CasthuZUÖB
´
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel/conv2d_5/Conv2D/CasthuZUÖB
™
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel/dense_2/MatMul/CasthuZUÖB
´
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel/conv2d_6/Conv2D/CasthuZUÖB
ƒ
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄb3gradient_tape/model/batch_normalization_8/Cast/CasthuZUÖB
≥
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄb"model/batch_normalization_8/Cast_1huZUÖB
´
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄbmodel/conv2d_9/Conv2D/CasthuZUÖB
ì
≈void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb,categorical_crossentropy/weighted_loss/valuehuZUÖB
Ò
≈void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb
div_no_nanhuZUÖB
Û
≈void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbdiv_no_nan_1huZUÖB
¨
≈void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbEgradient_tape/categorical_crossentropy/weighted_loss/value/div_no_nanhuZUÖB
¬
ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_inverse_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_inverse_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä28Ä@ÄHÄbtruedivhuZUÖB
ã
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbCast_1huZUÖB
ã
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbCast_6huZUÖB
∆
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbAcategorical_crossentropy/softmax_cross_entropy_with_logits/Cast_1huZUÖB
Ï
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbgcategorical_crossentropy/weighted_loss/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_CasthuZUÖB
ñ
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbêgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/Cast_2/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_CasthuZUÖB
≈
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb@gradient_tape/categorical_crossentropy/weighted_loss/Cast_1/CasthuZUÖB
¥
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb/gradient_tape/model/conv2d_12/BiasAdd/Cast/CasthuZUÖB
¥
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb/gradient_tape/model/conv2d_13/BiasAdd/Cast/CasthuZUÖB
¥
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb/gradient_tape/model/conv2d_15/BiasAdd/Cast/CasthuZUÖB
¥
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb/gradient_tape/model/conv2d_16/BiasAdd/Cast/CasthuZUÖB
¥
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb/gradient_tape/model/conv2d_17/BiasAdd/Cast/CasthuZUÖB
¥
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb/gradient_tape/model/conv2d_18/BiasAdd/Cast/CasthuZUÖB
¥
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb/gradient_tape/model/conv2d_19/BiasAdd/Cast/CasthuZUÖB
¥
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb/gradient_tape/model/conv2d_20/BiasAdd/Cast/CasthuZUÖB
¥
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb/gradient_tape/model/conv2d_22/BiasAdd/Cast/CasthuZUÖB
¥
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb/gradient_tape/model/conv2d_23/BiasAdd/Cast/CasthuZUÖB
≥
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb.gradient_tape/model/conv2d_3/BiasAdd/Cast/CasthuZUÖB
≤
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb-gradient_tape/model/conv2d_3/Conv2D/Cast/CasthuZUÖB
≥
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb.gradient_tape/model/conv2d_4/BiasAdd/Cast/CasthuZUÖB
≥
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb.gradient_tape/model/conv2d_5/BiasAdd/Cast/CasthuZUÖB
≥
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb.gradient_tape/model/conv2d_6/BiasAdd/Cast/CasthuZUÖB
≥
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb.gradient_tape/model/conv2d_7/BiasAdd/Cast/CasthuZUÖB
≥
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb.gradient_tape/model/conv2d_9/BiasAdd/Cast/CasthuZUÖB
≤
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb-gradient_tape/model/dense_2/BiasAdd/Cast/CasthuZUÖB
∞
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb+gradient_tape/model/dense/BiasAdd/Cast/CasthuZUÖB
±
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb,gradient_tape/model/dense_2/MatMul/Cast/CasthuZUÖB
•
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄb model/batch_normalization_8/CasthuZUÖB
˚
”void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbCast_2huZUÖB
≠
”void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb8categorical_crossentropy/weighted_loss/num_elements/CasthuZUÖB
´
”void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb6gradient_tape/model/batch_normalization_8/moments/CasthuZUÖB
´
”void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb6gradient_tape/model/batch_normalization_9/moments/CasthuZUÖB
≠
”void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb8gradient_tape/model/batch_normalization_9/moments/Cast_1huZUÖB
ô
’void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb"cond_1/then/_10/cond_1/Adam/Cast_1huZUÖB
¿
Òvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb-model/batch_normalization_8/AssignMovingAvg_1huZUÖB
¿
Òvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb-model/batch_normalization_9/AssignMovingAvg_1huZUÖB
ß
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbAdam/gradients/AddN_10huZUÖB
ß
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbAdam/gradients/AddN_11huZUÖB
¶
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbAdam/gradients/AddN_2huZUÖB
ò
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbAssignAddVariableOphuZUÖB
ö
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbAssignAddVariableOp_2huZUÖB
ö
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbAssignAddVariableOp_3huZUÖB
È
üvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long) *Ä28Ä@ÄHÄb(ArithmeticOptimizer/AddOpsRewrite_AddN_1huZUÖB
Ø
Ÿvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb4cond_1/then/_10/cond_1/Adam/Adam/AssignAddVariableOphuZUÖB
˜
õvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorEvalToOp<Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_log_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorEvalToOp<Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_log_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb:categorical_crossentropy/softmax_cross_entropy_with_logitshuZUÖB
Ö
€void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *Ä28Ä@ÄHÄbAll_10hu¶™¶B
Ö
€void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *Ä28Ä@ÄHÄbAll_20hu¶™¶B
Ö
€void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *Ä28Ä@ÄHÄbAll_26hu¶™¶B