
ô
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8Ó–°@Ó–°HÓ–°b<cond_1/then/_10/cond_1/Adam/Adam/update_56/ResourceApplyAdamhuZUÖB
˙
ï_ZN13cutlass_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi64ELi256ELi32EEENS4_51Conv2dWgradOutputGradientTileAccessIteratorAnalyticINS_11MatrixShapeILi64ELi32EEENS_6half_tENS_9transform29PitchLinearWarpRakedThreadMapINS_6layout16PitchLinearShapeILi64ELi32EEELi128ENSG_ILi8ELi4EEELi8EEEEENSD_11threadblock25RegularTileAccessIteratorISB_SC_NSF_40ColumnMajorTensorOpMultiplicandCongruousILi16ELi64EEELi1ESJ_Li16EEELNS_4arch14CacheOperation4KindE0ENS4_47Conv2dWgradActivationTileAccessIteratorAnalyticINSA_ILi32ELi256EEESC_NSE_INSG_ILi256ELi32EEELi128ESI_Li8EEEEENSM_ISU_SC_NSF_37RowMajorTensorOpMultiplicandCongruousILi16ELi64EEELi0ESW_Li16EEELSS_0ENS6_11threadblock9MmaPolicyINS6_4warp11MmaTensorOpINS7_ILi64ELi64ELi32EEESC_SO_SC_SZ_fNSF_8RowMajorENS13_17MmaTensorOpPolicyINSQ_3MmaINS7_ILi16ELi8ELi16EEELi32ESC_S16_SC_NSF_11ColumnMajorEfS16_NSQ_13OpMultiplyAddEEENSA_ILi1ELi1EEEEELi1ELb0EbEENSA_ILi0ELi0EEES1G_Li1EEELi4EbEENS_8epilogue11threadblock8EpilogueIS8_S1F_Li1ENS1K_22PredicatedTileIteratorINS1K_26OutputTileOptimalThreadMapINS1K_15OutputTileShapeILi256ELi8ELi1ELi1ELi1EEENS1O_ILi1ELi8ELi1ELi1ELi8EEELi128ELi4ELi32EEEfEENS1J_4warp24FragmentIteratorTensorOpIS15_S19_fNS_5ArrayIfLi4ELb1EEES16_EENS1T_20TileIteratorTensorOpIS15_S19_fS16_EENS1K_18SharedLoadIteratorINS1R_18CompactedThreadMapEfLi16EEENS1J_6thread17LinearCombinationIfLi4EffLNS_15FloatRoundStyleE2EEENSA_ILi0ELi8EEEEENS11_30GemmIdentityThreadblockSwizzleILi4EEELNS1_8OperatorE2ENS1_17Conv2dProblemSizeEEEEEvNT_6ParamsEˇ ÄÄ*Ä22)8¡¯é@¡¯éH¡¯éXb;gradient_tape/model_1/conv2d_35/Conv2D/Conv2DBackpropFilterh
õ
∂void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride>(cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride::Params)ı Ä¿*Ä2Ä8∏®◊@∏®◊H∏®◊Xb:gradient_tape/model_1/conv2d_35/Conv2D/Conv2DBackpropInputh
¿
|sm80_xmma_fprop_implicit_gemm_f16f16_f16f32_f32_nhwckrsc_nhwc_tilesize256x64x32_stage3_warpsize4x1x1_g1_tensor16x8x16_kernel˙ Ä‡*Ä2Ä8∏ê÷@∏ê÷H∏ê÷PXbmodel_1/conv2d_35/Conv2Dh
õ
∂void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride>(cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride::Params)ı Ä¿*Ä2Ä8´ÿå@´ÿåH´ÿåXb:gradient_tape/model_1/conv2d_41/Conv2D/Conv2DBackpropInputh
≠
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8¨∞å@¨∞åH¨∞åbAdam/gradients/AddN_31huZUÖB
w
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ä¿8´êÜ@´êÜH´êÜb.gradient_tape/dense_3/kernel/Regularizer/Mul_1huZUÖB
ﬂ
úvoid cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3>(cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3::Params)˛ Ä¿*Ä2Ä8™‡Ö@™‡ÖH™‡ÖXbmodel_1/conv2d_41/Conv2Dh
≈
ﬁvoid xmma_cudnn::gemm::kernel<xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, xmma_cudnn::Col, 256, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 3> >(xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, xmma_cudnn::Col, 256, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 3>::Params)Ó Ä¿*Ä2$	8™®Ç@™®ÇH™®ÇPXb;gradient_tape/model_1/conv2d_34/Conv2D/Conv2DBackpropFilterh
˙
ï_ZN13cutlass_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi64ELi256ELi32EEENS4_51Conv2dWgradOutputGradientTileAccessIteratorAnalyticINS_11MatrixShapeILi64ELi32EEENS_6half_tENS_9transform29PitchLinearWarpRakedThreadMapINS_6layout16PitchLinearShapeILi64ELi32EEELi128ENSG_ILi8ELi4EEELi8EEEEENSD_11threadblock25RegularTileAccessIteratorISB_SC_NSF_40ColumnMajorTensorOpMultiplicandCongruousILi16ELi64EEELi1ESJ_Li16EEELNS_4arch14CacheOperation4KindE0ENS4_47Conv2dWgradActivationTileAccessIteratorAnalyticINSA_ILi32ELi256EEESC_NSE_INSG_ILi256ELi32EEELi128ESI_Li8EEEEENSM_ISU_SC_NSF_37RowMajorTensorOpMultiplicandCongruousILi16ELi64EEELi0ESW_Li16EEELSS_0ENS6_11threadblock9MmaPolicyINS6_4warp11MmaTensorOpINS7_ILi64ELi64ELi32EEESC_SO_SC_SZ_fNSF_8RowMajorENS13_17MmaTensorOpPolicyINSQ_3MmaINS7_ILi16ELi8ELi16EEELi32ESC_S16_SC_NSF_11ColumnMajorEfS16_NSQ_13OpMultiplyAddEEENSA_ILi1ELi1EEEEELi1ELb0EbEENSA_ILi0ELi0EEES1G_Li1EEELi4EbEENS_8epilogue11threadblock8EpilogueIS8_S1F_Li1ENS1K_22PredicatedTileIteratorINS1K_26OutputTileOptimalThreadMapINS1K_15OutputTileShapeILi256ELi8ELi1ELi1ELi1EEENS1O_ILi1ELi8ELi1ELi1ELi8EEELi128ELi4ELi32EEEfEENS1J_4warp24FragmentIteratorTensorOpIS15_S19_fNS_5ArrayIfLi4ELb1EEES16_EENS1T_20TileIteratorTensorOpIS15_S19_fS16_EENS1K_18SharedLoadIteratorINS1R_18CompactedThreadMapEfLi16EEENS1J_6thread17LinearCombinationIfLi4EffLNS_15FloatRoundStyleE2EEENSA_ILi0ELi8EEEEENS11_30GemmIdentityThreadblockSwizzleILi4EEELNS1_8OperatorE2ENS1_17Conv2dProblemSizeEEEEEvNT_6ParamsEˇ ÄÄ*Ä2
8©Ä¸@©Ä¸H©Ä¸Xb;gradient_tape/model_1/conv2d_41/Conv2D/Conv2DBackpropFilterh
ﬂ
úvoid cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3>(cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3::Params)˛ Ä¿*Ä2Ä8©¯¯@©¯¯H©¯¯Xbmodel_1/conv2d_34/Conv2Dh
õ
∂void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride>(cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride::Params)ı Ä¿*Ä2Ä8©®¯@©®¯H©®¯Xb:gradient_tape/model_1/conv2d_34/Conv2D/Conv2DBackpropInputh
ﬂ
úvoid cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3>(cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3::Params)˛ Ä¿*Ä2Ä8¢ÿ—@¢ÿ—H¢ÿ—Xbmodel_1/conv2d_38/Conv2Dh
˙
ï_ZN13cutlass_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi64ELi256ELi32EEENS4_51Conv2dWgradOutputGradientTileAccessIteratorAnalyticINS_11MatrixShapeILi64ELi32EEENS_6half_tENS_9transform29PitchLinearWarpRakedThreadMapINS_6layout16PitchLinearShapeILi64ELi32EEELi128ENSG_ILi8ELi4EEELi8EEEEENSD_11threadblock25RegularTileAccessIteratorISB_SC_NSF_40ColumnMajorTensorOpMultiplicandCongruousILi16ELi64EEELi1ESJ_Li16EEELNS_4arch14CacheOperation4KindE0ENS4_47Conv2dWgradActivationTileAccessIteratorAnalyticINSA_ILi32ELi256EEESC_NSE_INSG_ILi256ELi32EEELi128ESI_Li8EEEEENSM_ISU_SC_NSF_37RowMajorTensorOpMultiplicandCongruousILi16ELi64EEELi0ESW_Li16EEELSS_0ENS6_11threadblock9MmaPolicyINS6_4warp11MmaTensorOpINS7_ILi64ELi64ELi32EEESC_SO_SC_SZ_fNSF_8RowMajorENS13_17MmaTensorOpPolicyINSQ_3MmaINS7_ILi16ELi8ELi16EEELi32ESC_S16_SC_NSF_11ColumnMajorEfS16_NSQ_13OpMultiplyAddEEENSA_ILi1ELi1EEEEELi1ELb0EbEENSA_ILi0ELi0EEES1G_Li1EEELi4EbEENS_8epilogue11threadblock8EpilogueIS8_S1F_Li1ENS1K_22PredicatedTileIteratorINS1K_26OutputTileOptimalThreadMapINS1K_15OutputTileShapeILi256ELi8ELi1ELi1ELi1EEENS1O_ILi1ELi8ELi1ELi1ELi8EEELi128ELi4ELi32EEEfEENS1J_4warp24FragmentIteratorTensorOpIS15_S19_fNS_5ArrayIfLi4ELb1EEES16_EENS1T_20TileIteratorTensorOpIS15_S19_fS16_EENS1K_18SharedLoadIteratorINS1R_18CompactedThreadMapEfLi16EEENS1J_6thread17LinearCombinationIfLi4EffLNS_15FloatRoundStyleE2EEENSA_ILi0ELi8EEEEENS11_30GemmIdentityThreadblockSwizzleILi4EEELNS1_8OperatorE2ENS1_17Conv2dProblemSizeEEEEEvNT_6ParamsEˇ ÄÄ*Ä2
8°ÿ @°ÿ H°ÿ Xb;gradient_tape/model_1/conv2d_38/Conv2D/Conv2DBackpropFilterh
ﬂ
úvoid cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3>(cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3::Params)˛ Ä¿*Ä2Ä8†∞¡@†∞¡H†∞¡Xbmodel_1/conv2d_46/Conv2Dh
õ
∂void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride>(cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride::Params)ı Ä¿*Ä2Ä8üàº@üàºHüàºXb:gradient_tape/model_1/conv2d_38/Conv2D/Conv2DBackpropInputh
≈
ﬁvoid xmma_cudnn::gemm::kernel<xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, xmma_cudnn::Col, 256, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 3> >(xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, xmma_cudnn::Col, 256, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 3>::Params)Ó Ä¿*Ä2Z8ù†≥@ù†≥Hù†≥PXb;gradient_tape/model_1/conv2d_46/Conv2D/Conv2DBackpropFilterh
≥
Œvoid cudnn::ops::pooling_bw_kernel_max<__half, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, false>(cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor) Ä *Ä2Ä 8ù¯±@ù¯±Hù¯±b9gradient_tape/model_1/max_pooling2d_5/MaxPool/MaxPoolGradhu  »B
m
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ä¿8ùà±@ùà±Hùà±b!dense_3/kernel/Regularizer/SquarehuZUÖB
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ä¿8ùÿÆ@ùÿÆHùÿÆb,gradient_tape/dense_3/kernel/Regularizer/MulhuZUÖB
O
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ä¿8úÿÆ@úÿÆHúÿÆbmul_58huZUÖB
π
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8ù∞≠@ù∞≠Hù∞≠b.gradient_tape/model_1/dense_3/MatMul/Cast/CasthuZUÖB
≤
Œvoid cudnn::ops::pooling_bw_kernel_max<__half, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, false>(cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor) Ä *Ä2   8úË©@úË©HúË©b9gradient_tape/model_1/max_pooling2d_4/MaxPool/MaxPoolGradhu  »B
≤
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8ôÿö@ôÿöHôÿöbmodel_1/dense_3/MatMul/CasthuZUÖB
õ
∂void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride>(cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride::Params)ı Ä¿*Ä2Ä8ô»ö@ô»öHô»öXb:gradient_tape/model_1/conv2d_46/Conv2D/Conv2DBackpropInputh
ø
|sm80_xmma_fprop_implicit_gemm_f16f16_f16f32_f32_nhwckrsc_nhwc_tilesize256x64x32_stage3_warpsize4x1x1_g1_tensor16x8x16_kernel˙ Ä‡*Ä2 8ñ®Ü@ñ®ÜHñ®ÜPXbmodel_1/conv2d_47/Conv2Dh
˙
ï_ZN13cutlass_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi64ELi256ELi32EEENS4_51Conv2dWgradOutputGradientTileAccessIteratorAnalyticINS_11MatrixShapeILi64ELi32EEENS_6half_tENS_9transform29PitchLinearWarpRakedThreadMapINS_6layout16PitchLinearShapeILi64ELi32EEELi128ENSG_ILi8ELi4EEELi8EEEEENSD_11threadblock25RegularTileAccessIteratorISB_SC_NSF_40ColumnMajorTensorOpMultiplicandCongruousILi16ELi64EEELi1ESJ_Li16EEELNS_4arch14CacheOperation4KindE0ENS4_47Conv2dWgradActivationTileAccessIteratorAnalyticINSA_ILi32ELi256EEESC_NSE_INSG_ILi256ELi32EEELi128ESI_Li8EEEEENSM_ISU_SC_NSF_37RowMajorTensorOpMultiplicandCongruousILi16ELi64EEELi0ESW_Li16EEELSS_0ENS6_11threadblock9MmaPolicyINS6_4warp11MmaTensorOpINS7_ILi64ELi64ELi32EEESC_SO_SC_SZ_fNSF_8RowMajorENS13_17MmaTensorOpPolicyINSQ_3MmaINS7_ILi16ELi8ELi16EEELi32ESC_S16_SC_NSF_11ColumnMajorEfS16_NSQ_13OpMultiplyAddEEENSA_ILi1ELi1EEEEELi1ELb0EbEENSA_ILi0ELi0EEES1G_Li1EEELi4EbEENS_8epilogue11threadblock8EpilogueIS8_S1F_Li1ENS1K_22PredicatedTileIteratorINS1K_26OutputTileOptimalThreadMapINS1K_15OutputTileShapeILi256ELi8ELi1ELi1ELi1EEENS1O_ILi1ELi8ELi1ELi1ELi8EEELi128ELi4ELi32EEEfEENS1J_4warp24FragmentIteratorTensorOpIS15_S19_fNS_5ArrayIfLi4ELb1EEES16_EENS1T_20TileIteratorTensorOpIS15_S19_fS16_EENS1K_18SharedLoadIteratorINS1R_18CompactedThreadMapEfLi16EEENS1J_6thread17LinearCombinationIfLi4EffLNS_15FloatRoundStyleE2EEENSA_ILi0ELi8EEEEENS11_30GemmIdentityThreadblockSwizzleILi4EEELNS1_8OperatorE2ENS1_17Conv2dProblemSizeEEEEEvNT_6ParamsEˇ ÄÄ*Ä2 8ñòÖ@ñòÖHñòÖXb;gradient_tape/model_1/conv2d_47/Conv2D/Conv2DBackpropFilterh
ò
∂void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride>(cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride::Params)ı Ä¿*Ä2Ä8íËk@íËkHíËkXb:gradient_tape/model_1/conv2d_47/Conv2D/Conv2DBackpropInputh
U
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2ÄÄ8ë»i@ë»iHë»ibIsFinite_56hu  »B
’
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8ë¿i@ë¿iHë¿ib-gradient_tape/dense_3/kernel/Regularizer/TilehuZUÖB
¬
ﬁvoid xmma_cudnn::gemm::kernel<xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 32, 2, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 32, 2, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 32, 2, 2, 1, 1>, 16, xmma_cudnn::Col, 128, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 32, 2, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 32, 2, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 4> >(xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 32, 2, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 32, 2, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 32, 2, 2, 1, 1>, 16, xmma_cudnn::Col, 128, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 32, 2, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 32, 2, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 4>::Params)Ù ÄÄ*Ä2	8ë†g@ë†gHë†gPXb;gradient_tape/model_1/conv2d_32/Conv2D/Conv2DBackpropFilterh
ò
∂void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride>(cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride::Params)ı Ä¿*Ä2Ä8ê®c@ê®cHê®cXb:gradient_tape/model_1/conv2d_40/Conv2D/Conv2DBackpropInputh
˜
ï_ZN13cutlass_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi64ELi256ELi32EEENS4_51Conv2dWgradOutputGradientTileAccessIteratorAnalyticINS_11MatrixShapeILi64ELi32EEENS_6half_tENS_9transform29PitchLinearWarpRakedThreadMapINS_6layout16PitchLinearShapeILi64ELi32EEELi128ENSG_ILi8ELi4EEELi8EEEEENSD_11threadblock25RegularTileAccessIteratorISB_SC_NSF_40ColumnMajorTensorOpMultiplicandCongruousILi16ELi64EEELi1ESJ_Li16EEELNS_4arch14CacheOperation4KindE0ENS4_47Conv2dWgradActivationTileAccessIteratorAnalyticINSA_ILi32ELi256EEESC_NSE_INSG_ILi256ELi32EEELi128ESI_Li8EEEEENSM_ISU_SC_NSF_37RowMajorTensorOpMultiplicandCongruousILi16ELi64EEELi0ESW_Li16EEELSS_0ENS6_11threadblock9MmaPolicyINS6_4warp11MmaTensorOpINS7_ILi64ELi64ELi32EEESC_SO_SC_SZ_fNSF_8RowMajorENS13_17MmaTensorOpPolicyINSQ_3MmaINS7_ILi16ELi8ELi16EEELi32ESC_S16_SC_NSF_11ColumnMajorEfS16_NSQ_13OpMultiplyAddEEENSA_ILi1ELi1EEEEELi1ELb0EbEENSA_ILi0ELi0EEES1G_Li1EEELi4EbEENS_8epilogue11threadblock8EpilogueIS8_S1F_Li1ENS1K_22PredicatedTileIteratorINS1K_26OutputTileOptimalThreadMapINS1K_15OutputTileShapeILi256ELi8ELi1ELi1ELi1EEENS1O_ILi1ELi8ELi1ELi1ELi8EEELi128ELi4ELi32EEEfEENS1J_4warp24FragmentIteratorTensorOpIS15_S19_fNS_5ArrayIfLi4ELb1EEES16_EENS1T_20TileIteratorTensorOpIS15_S19_fS16_EENS1K_18SharedLoadIteratorINS1R_18CompactedThreadMapEfLi16EEENS1J_6thread17LinearCombinationIfLi4EffLNS_15FloatRoundStyleE2EEENSA_ILi0ELi8EEEEENS11_30GemmIdentityThreadblockSwizzleILi4EEELNS1_8OperatorE2ENS1_17Conv2dProblemSizeEEEEEvNT_6ParamsEˇ ÄÄ*Ä2
8ê¿a@ê¿aHê¿aXb;gradient_tape/model_1/conv2d_40/Conv2D/Conv2DBackpropFilterh
‹
úvoid cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3>(cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3::Params)˛ Ä¿*Ä2Ä8èòa@èòaHèòaXbmodel_1/conv2d_40/Conv2Dh
‹
úvoid cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3>(cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3::Params)˛ Ä¿*Ä2Ä8è®^@è®^Hè®^Xbmodel_1/conv2d_32/Conv2Dh
≤
Œvoid xmma_cudnn::gemm::kernel<xmma_cudnn::implicit_gemm::dgrad::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_a_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 16, false, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 16, xmma_cudnn::Row, 32, 256> >, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_c_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, 16, xmma_cudnn::Fragment_c<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, false>, false>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 3> >(xmma_cudnn::implicit_gemm::dgrad::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_a_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 16, false, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 16, xmma_cudnn::Row, 32, 256> >, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_c_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, 16, xmma_cudnn::Fragment_c<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, false>, false>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 3>::Params)ˇ Ä‡*Ä2Ä8è»]@è»]Hè»]PXb:gradient_tape/model_1/conv2d_32/Conv2D/Conv2DBackpropInputh
¡
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2ú8éËR@éËRHéËRbdense_3/kernel/Regularizer/Sumhu  »B
ó
∂void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride>(cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride::Params)ı Ä¿*Ä2@8çËP@çËPHçËPXb:gradient_tape/model_1/conv2d_44/Conv2D/Conv2DBackpropInputh
‹
úvoid cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3>(cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3::Params)˛ Ä¿*Ä2Ä8çÄM@çÄMHçÄMXbmodel_1/conv2d_37/Conv2Dh
˜
ï_ZN13cutlass_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi64ELi256ELi32EEENS4_51Conv2dWgradOutputGradientTileAccessIteratorAnalyticINS_11MatrixShapeILi64ELi32EEENS_6half_tENS_9transform29PitchLinearWarpRakedThreadMapINS_6layout16PitchLinearShapeILi64ELi32EEELi128ENSG_ILi8ELi4EEELi8EEEEENSD_11threadblock25RegularTileAccessIteratorISB_SC_NSF_40ColumnMajorTensorOpMultiplicandCongruousILi16ELi64EEELi1ESJ_Li16EEELNS_4arch14CacheOperation4KindE0ENS4_47Conv2dWgradActivationTileAccessIteratorAnalyticINSA_ILi32ELi256EEESC_NSE_INSG_ILi256ELi32EEELi128ESI_Li8EEEEENSM_ISU_SC_NSF_37RowMajorTensorOpMultiplicandCongruousILi16ELi64EEELi0ESW_Li16EEELSS_0ENS6_11threadblock9MmaPolicyINS6_4warp11MmaTensorOpINS7_ILi64ELi64ELi32EEESC_SO_SC_SZ_fNSF_8RowMajorENS13_17MmaTensorOpPolicyINSQ_3MmaINS7_ILi16ELi8ELi16EEELi32ESC_S16_SC_NSF_11ColumnMajorEfS16_NSQ_13OpMultiplyAddEEENSA_ILi1ELi1EEEEELi1ELb0EbEENSA_ILi0ELi0EEES1G_Li1EEELi4EbEENS_8epilogue11threadblock8EpilogueIS8_S1F_Li1ENS1K_22PredicatedTileIteratorINS1K_26OutputTileOptimalThreadMapINS1K_15OutputTileShapeILi256ELi8ELi1ELi1ELi1EEENS1O_ILi1ELi8ELi1ELi1ELi8EEELi128ELi4ELi32EEEfEENS1J_4warp24FragmentIteratorTensorOpIS15_S19_fNS_5ArrayIfLi4ELb1EEES16_EENS1T_20TileIteratorTensorOpIS15_S19_fS16_EENS1K_18SharedLoadIteratorINS1R_18CompactedThreadMapEfLi16EEENS1J_6thread17LinearCombinationIfLi4EffLNS_15FloatRoundStyleE2EEENSA_ILi0ELi8EEEEENS11_30GemmIdentityThreadblockSwizzleILi4EEELNS1_8OperatorE2ENS1_17Conv2dProblemSizeEEEEEvNT_6ParamsEˇ ÄÄ*Ä2
8å»I@å»IHå»IXb;gradient_tape/model_1/conv2d_37/Conv2D/Conv2DBackpropFilterh
˜
ï_ZN13cutlass_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi64ELi256ELi32EEENS4_51Conv2dWgradOutputGradientTileAccessIteratorAnalyticINS_11MatrixShapeILi64ELi32EEENS_6half_tENS_9transform29PitchLinearWarpRakedThreadMapINS_6layout16PitchLinearShapeILi64ELi32EEELi128ENSG_ILi8ELi4EEELi8EEEEENSD_11threadblock25RegularTileAccessIteratorISB_SC_NSF_40ColumnMajorTensorOpMultiplicandCongruousILi16ELi64EEELi1ESJ_Li16EEELNS_4arch14CacheOperation4KindE0ENS4_47Conv2dWgradActivationTileAccessIteratorAnalyticINSA_ILi32ELi256EEESC_NSE_INSG_ILi256ELi32EEELi128ESI_Li8EEEEENSM_ISU_SC_NSF_37RowMajorTensorOpMultiplicandCongruousILi16ELi64EEELi0ESW_Li16EEELSS_0ENS6_11threadblock9MmaPolicyINS6_4warp11MmaTensorOpINS7_ILi64ELi64ELi32EEESC_SO_SC_SZ_fNSF_8RowMajorENS13_17MmaTensorOpPolicyINSQ_3MmaINS7_ILi16ELi8ELi16EEELi32ESC_S16_SC_NSF_11ColumnMajorEfS16_NSQ_13OpMultiplyAddEEENSA_ILi1ELi1EEEEELi1ELb0EbEENSA_ILi0ELi0EEES1G_Li1EEELi4EbEENS_8epilogue11threadblock8EpilogueIS8_S1F_Li1ENS1K_22PredicatedTileIteratorINS1K_26OutputTileOptimalThreadMapINS1K_15OutputTileShapeILi256ELi8ELi1ELi1ELi1EEENS1O_ILi1ELi8ELi1ELi1ELi8EEELi128ELi4ELi32EEEfEENS1J_4warp24FragmentIteratorTensorOpIS15_S19_fNS_5ArrayIfLi4ELb1EEES16_EENS1T_20TileIteratorTensorOpIS15_S19_fS16_EENS1K_18SharedLoadIteratorINS1R_18CompactedThreadMapEfLi16EEENS1J_6thread17LinearCombinationIfLi4EffLNS_15FloatRoundStyleE2EEENSA_ILi0ELi8EEEEENS11_30GemmIdentityThreadblockSwizzleILi4EEELNS1_8OperatorE2ENS1_17Conv2dProblemSizeEEEEEvNT_6ParamsEˇ ÄÄ*Ä2	8åòI@åòIHåòIXb;gradient_tape/model_1/conv2d_44/Conv2D/Conv2DBackpropFilterh
¬
ﬁvoid xmma_cudnn::gemm::kernel<xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, xmma_cudnn::Col, 256, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 3> >(xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, xmma_cudnn::Col, 256, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 3>::Params)Ó Ä¿*Ä2-8ã–G@ã–GHã–GPXb;gradient_tape/model_1/conv2d_43/Conv2D/Conv2DBackpropFilterh
ò
∂void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride>(cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride::Params)ı Ä¿*Ä2Ä8ã¿E@ã¿EHã¿EXb:gradient_tape/model_1/conv2d_37/Conv2D/Conv2DBackpropInputh
≤
Œvoid xmma_cudnn::gemm::kernel<xmma_cudnn::implicit_gemm::dgrad::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_a_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 16, false, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 16, xmma_cudnn::Row, 32, 256> >, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_c_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, 16, xmma_cudnn::Fragment_c<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, false>, false>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 3> >(xmma_cudnn::implicit_gemm::dgrad::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_a_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 16, false, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 16, xmma_cudnn::Row, 32, 256> >, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_c_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, 16, xmma_cudnn::Fragment_c<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, false>, false>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 3>::Params)ˇ Ä‡*Ä2Ä8ãêD@ãêDHãêDPXb:gradient_tape/model_1/conv2d_31/Conv2D/Conv2DBackpropInputh
‹
úvoid cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3>(cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3::Params)˛ Ä¿*Ä2Ä8ã¯C@ã¯CHã¯CXbmodel_1/conv2d_31/Conv2Dh
º
|sm80_xmma_fprop_implicit_gemm_f16f16_f16f32_f32_nhwckrsc_nhwc_tilesize256x64x32_stage3_warpsize4x1x1_g1_tensor16x8x16_kernel˙ Ä‡*Ä2 8ã»C@ã»CHã»CPXbmodel_1/conv2d_44/Conv2Dh
¬
ﬁvoid xmma_cudnn::gemm::kernel<xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, xmma_cudnn::Col, 256, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 3> >(xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, xmma_cudnn::Col, 256, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 3>::Params)Ó Ä¿*Ä2		8ã»A@ã»AHã»APXb;gradient_tape/model_1/conv2d_31/Conv2D/Conv2DBackpropFilterh
ê
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2ÄÄ8â®;@É¿HÉ®bAgradient_tape/model_1/batch_normalization_11/FusedBatchNormGradV3hu  »B
ó
∂void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride>(cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride::Params)ı Ä¿*Ä2@8ä–:@ä–:Hä–:Xb:gradient_tape/model_1/conv2d_43/Conv2D/Conv2DBackpropInputh
ê
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2ÄÄ8ä–:@É¿HÑ»bAgradient_tape/model_1/batch_normalization_12/FusedBatchNormGradV3hu  »B
ê
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2ÄÄ8â∞:@É∏HÉ¿bAgradient_tape/model_1/batch_normalization_13/FusedBatchNormGradV3hu  »B
Ì
Évoid cudnn::bn_bw_1C11_kernel_new<__half, float, float2, 512, true, 1>(float, float, float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, float const*, float*, float*, float const*, float const*, float)(ê*Ä2Ä8ä¯8@ä¯8Hä¯8bAgradient_tape/model_1/batch_normalization_11/FusedBatchNormGradV3hu  »B
Ì
Évoid cudnn::bn_bw_1C11_kernel_new<__half, float, float2, 512, true, 1>(float, float, float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, float const*, float*, float*, float const*, float const*, float)(ê*Ä2Ä8âê7@âê7Hâê7bAgradient_tape/model_1/batch_normalization_12/FusedBatchNormGradV3hu  »B
∞
Œvoid cudnn::ops::pooling_bw_kernel_max<__half, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, false>(cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor) Ä *Ä2† 8âË6@âË6HâË6b9gradient_tape/model_1/max_pooling2d_6/MaxPool/MaxPoolGradhu  »B
Ì
Évoid cudnn::bn_bw_1C11_kernel_new<__half, float, float2, 512, true, 1>(float, float, float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, float const*, float*, float*, float const*, float const*, float)(ê*Ä2Ä8â®6@â®6Hâ®6bAgradient_tape/model_1/batch_normalization_13/FusedBatchNormGradV3hu  »B
‹
úvoid cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3>(cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3::Params)˛ Ä¿*Ä2Ä8àÄ1@àÄ1HàÄ1Xbmodel_1/conv2d_43/Conv2Dh
}
-ampere_fp16_s1688gemm_fp16_256x64_ldg8_f2f_ntÍÄ¿*Ä2Ä8áË-@áË-HáË-b&gradient_tape/model_1/dense_3/MatMul_1hugUÖA
z
:ampere_fp16_s16816gemm_fp16_128x64_ldg8_f2f_stages_64x4_nnñÄÄ ÄÄ*Ä28à¯,@à¯,Hà¯,Xbmodel_1/dense_3/MatMulh
â
:ampere_fp16_s16816gemm_fp16_128x64_ldg8_f2f_stages_64x4_tnûÄÄ ÄÄ*Ä2Ä	8áò,@áò,Háò,Xb$gradient_tape/model_1/dense_3/MatMulh
∞
Œvoid cudnn::ops::pooling_bw_kernel_max<__half, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, false>(cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor) Ä *Ä2† 8á‡)@á‡)Há‡)b9gradient_tape/model_1/max_pooling2d_7/MaxPool/MaxPoolGradhu  »B
˛
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2ÄÄ8Ü†'@É»HÉÿb/model_1/batch_normalization_13/FusedBatchNormV3hu  »B
˛
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2ÄÄ8áê'@É¿HÑ–b/model_1/batch_normalization_12/FusedBatchNormV3hu  »B
˛
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2ÄÄ8á&@É∏HÑ∏b/model_1/batch_normalization_11/FusedBatchNormV3hu  »B
Ÿ

ü
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ü¯%@Ü¯%HÜ¯%bAdam/gradients/AddN_8huZUÖB
∞
ÿvoid cudnn::bn_fw_tr_1C11_kernel_NCHW<__half, float, 512, true, 1>(cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(ê*Ä2Ä8Ö¿ @Ö¿ HÖ¿ b/model_1/batch_normalization_12/FusedBatchNormV3hu  »B
∞
ÿvoid cudnn::bn_fw_tr_1C11_kernel_NCHW<__half, float, 512, true, 1>(cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(ê*Ä2Ä8Ü∏ @Ü∏ HÜ∏ b/model_1/batch_normalization_13/FusedBatchNormV3hu  »B
˜
ï_ZN13cutlass_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi64ELi256ELi32EEENS4_51Conv2dWgradOutputGradientTileAccessIteratorAnalyticINS_11MatrixShapeILi64ELi32EEENS_6half_tENS_9transform29PitchLinearWarpRakedThreadMapINS_6layout16PitchLinearShapeILi64ELi32EEELi128ENSG_ILi8ELi4EEELi8EEEEENSD_11threadblock25RegularTileAccessIteratorISB_SC_NSF_40ColumnMajorTensorOpMultiplicandCongruousILi16ELi64EEELi1ESJ_Li16EEELNS_4arch14CacheOperation4KindE0ENS4_47Conv2dWgradActivationTileAccessIteratorAnalyticINSA_ILi32ELi256EEESC_NSE_INSG_ILi256ELi32EEELi128ESI_Li8EEEEENSM_ISU_SC_NSF_37RowMajorTensorOpMultiplicandCongruousILi16ELi64EEELi0ESW_Li16EEELSS_0ENS6_11threadblock9MmaPolicyINS6_4warp11MmaTensorOpINS7_ILi64ELi64ELi32EEESC_SO_SC_SZ_fNSF_8RowMajorENS13_17MmaTensorOpPolicyINSQ_3MmaINS7_ILi16ELi8ELi16EEELi32ESC_S16_SC_NSF_11ColumnMajorEfS16_NSQ_13OpMultiplyAddEEENSA_ILi1ELi1EEEEELi1ELb0EbEENSA_ILi0ELi0EEES1G_Li1EEELi4EbEENS_8epilogue11threadblock8EpilogueIS8_S1F_Li1ENS1K_22PredicatedTileIteratorINS1K_26OutputTileOptimalThreadMapINS1K_15OutputTileShapeILi256ELi8ELi1ELi1ELi1EEENS1O_ILi1ELi8ELi1ELi1ELi8EEELi128ELi4ELi32EEEfEENS1J_4warp24FragmentIteratorTensorOpIS15_S19_fNS_5ArrayIfLi4ELb1EEES16_EENS1T_20TileIteratorTensorOpIS15_S19_fS16_EENS1K_18SharedLoadIteratorINS1R_18CompactedThreadMapEfLi16EEENS1J_6thread17LinearCombinationIfLi4EffLNS_15FloatRoundStyleE2EEENSA_ILi0ELi8EEEEENS11_30GemmIdentityThreadblockSwizzleILi4EEELNS1_8OperatorE2ENS1_17Conv2dProblemSizeEEEEEvNT_6ParamsEˇ ÄÄ*Ä2)8Ö∏@Ö∏HÖ∏Xb;gradient_tape/model_1/conv2d_29/Conv2D/Conv2DBackpropFilterh
‡
ùvoid convolve_common_engine_float_NHWC<__half, __half, 128, 5, 5, 3, 3, 3, true, false, false, false, false>(int, int, int, __half const*, __half const*, int, __half*, conv_kernel_common_params, unsigned long long, unsigned long, float, float, bool, __half const*, __half const*, bool)PÄ*2ÄÄ8Ö∏@Ö∏HÖ∏Xbmodel_1/conv2d_29/Conv2Dhu  HB
∞
ÿvoid cudnn::bn_fw_tr_1C11_kernel_NCHW<__half, float, 512, true, 1>(cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(ê*Ä2Ä8Öÿ@ÖÿHÖÿb/model_1/batch_normalization_11/FusedBatchNormV3hu  »B
Ñ
√void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*Ä2{8Ö–@Å†HÉàbmodel_1/concatenate_9/concathuZUÖB
∫
htensorflow::functor::ReluGradHalfKernelVector(Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2ÄÄ8Öà@ÖàHÖàb,gradient_tape/model_1/activation_13/ReluGradhu  »B
∫
htensorflow::functor::ReluGradHalfKernelVector(Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2ÄÄ8ÖÄ@ÖÄHÖÄb,gradient_tape/model_1/activation_12/ReluGradhu  »B
∫
htensorflow::functor::ReluGradHalfKernelVector(Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2ÄÄ8Ö@ÖHÖb,gradient_tape/model_1/activation_14/ReluGradhu  »B
Ö
√void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*Ä2{8Ö»@Å†HÇàbmodel_1/concatenate_11/concathuZUÖB
Ö
√void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*Ä2{8Ñ®@ÅêHÇÄbmodel_1/concatenate_10/concathuZUÖB
ñ
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8ÑÄ@ÑÄHÑÄb<cond_1/then/_10/cond_1/Adam/Adam/update_50/ResourceApplyAdamhuZUÖB
†
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_tn‚ÄÄ ÄÄ*Ä2Ä8É¿@É¿HÉ¿Xb:gradient_tape/model_1/conv2d_33/Conv2D/Conv2DBackpropInputh
å
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2ú8Ñà@ÑàHÑàbAll_56hu  »B
}
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_nnﬁÄÄ ÄÄ*Ä2@8Ñ∞@Ñ∞HÑ∞Xbmodel_1/conv2d_45/Conv2Dh
¶	
Ávoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*Ä2{8É®@É®HÉ®bmodel_1/activation_12/ReluhuZUÖB
¶	
Ávoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*Ä2{8É®@É®HÉ®bmodel_1/activation_14/ReluhuZUÖB
¶	
Ávoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*Ä2{8É†@É†HÉ†bmodel_1/activation_13/ReluhuZUÖB
¨
Àvoid cutlass_cudnn::Kernel<cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4> >(cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4>::Params)6* 2 28É∏@É∏HÉ∏Xb;gradient_tape/model_1/conv2d_35/Conv2D/Conv2DBackpropFilterhu  ñB
ü
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_tn‚ÄÄ ÄÄ*Ä2
@8Éà@ÉàHÉàXb:gradient_tape/model_1/conv2d_45/Conv2D/Conv2DBackpropInputh
ê
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2Ä†8Éÿ@ÅòHÅ®bAgradient_tape/model_1/batch_normalization_14/FusedBatchNormGradV3hu  »B
ê
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2Ä†8Éÿ@ÅêHÅ®bAgradient_tape/model_1/batch_normalization_15/FusedBatchNormGradV3hu  »B
†
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_64x3_ntÊÄÄ ÄÄ*Ä2
8É¿@É¿HÉ¿Xb;gradient_tape/model_1/conv2d_45/Conv2D/Conv2DBackpropFilterh
ñ
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8É¿@É¿HÉ¿b<cond_1/then/_10/cond_1/Adam/Adam/update_52/ResourceApplyAdamhuZUÖB
ê
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2Äê8É@Å»HÅÿbAgradient_tape/model_1/batch_normalization_17/FusedBatchNormGradV3hu  »B
~
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_nnﬁÄÄ ÄÄ*Ä2Ä8É¿@É¿HÉ¿Xbmodel_1/conv2d_33/Conv2Dh
ô
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Çê@ÇêHÇêb+gradient_tape/model_1/concatenate_9/Slice_2huZUÖB
ö
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Çà@ÇàHÇàb,gradient_tape/model_1/concatenate_10/Slice_1huZUÖB
Ì
Évoid cudnn::bn_bw_1C11_kernel_new<__half, float, float2, 128, true, 1>(float, float, float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, float const*, float*, float*, float const*, float const*, float)(ê*Ä2Ä8ÉË@ÉËHÉËbAgradient_tape/model_1/batch_normalization_15/FusedBatchNormGradV3hu  »B
ö
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8ÇË@ÇËHÇËb,gradient_tape/model_1/concatenate_11/Slice_1huZUÖB
Ì
Évoid cudnn::bn_bw_1C11_kernel_new<__half, float, float2, 128, true, 1>(float, float, float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, float const*, float*, float*, float const*, float const*, float)(ê*Ä2Ä8ÇË@ÇËHÇËbAgradient_tape/model_1/batch_normalization_14/FusedBatchNormGradV3hu  »B
†
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_64x3_ntÊÄÄ ÄÄ*Ä28É‡@É‡HÉ‡Xb;gradient_tape/model_1/conv2d_33/Conv2D/Conv2DBackpropFilterh
©
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Ç®@Ç®HÇ®bmodel_1/conv2d_29/BiasAddhuZUÖB
©
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Ç®@Ç®HÇ®bmodel_1/conv2d_31/BiasAddhuZUÖB
©
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Ç†@Ç†HÇ†bmodel_1/conv2d_34/BiasAddhuZUÖB
˛
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2Ä†8Ç»@Å†HÅ®b/model_1/batch_normalization_15/FusedBatchNormV3hu  »B
“
åvoid pooling_fw_kernel_max_nhwc<__half, float, 0, (cudnnNanPropagation_t)0, false>(PoolingParams, __half*, __half const*, float, float, int)*Ä2ÄÄ8Ç®@Ç®HÇ®bmodel_1/max_pooling2d_5/MaxPoolhu  »B
“
åvoid pooling_fw_kernel_max_nhwc<__half, float, 0, (cudnnNanPropagation_t)0, false>(PoolingParams, __half*, __half const*, float, float, int)*Ä2ÄÄ8Çà@ÇàHÇàbmodel_1/max_pooling2d_4/MaxPoolhu  »B
¨
Àvoid cutlass_cudnn::Kernel<cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4> >(cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4>::Params)6* 2@}8Ç@ÇHÇXb;gradient_tape/model_1/conv2d_47/Conv2D/Conv2DBackpropFilterhu  ñB
–
Ô_ZN13cutlass_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi64ELi64ELi32EEENS4_51Conv2dWgradOutputGradientTileAccessIteratorAnalyticINS_11MatrixShapeILi64ELi32EEENS_6half_tENS_9transform29PitchLinearWarpRakedThreadMapINS_6layout16PitchLinearShapeILi64ELi32EEELi128ENSG_ILi8ELi4EEELi8EEEEENSD_11threadblock25RegularTileAccessIteratorISB_SC_NSF_40ColumnMajorTensorOpMultiplicandCongruousILi16ELi64EEELi1ESJ_Li16EEELNS_4arch14CacheOperation4KindE0ENS4_47Conv2dWgradActivationTileAccessIteratorAnalyticINSA_ILi32ELi64EEESC_SJ_EENSM_ISU_SC_NSF_37RowMajorTensorOpMultiplicandCongruousILi16ELi64EEELi0ESJ_Li16EEELSS_0ENS6_11threadblock9MmaPolicyINS6_4warp11MmaTensorOpINS7_ILi32ELi32ELi32EEESC_SO_SC_SX_fNSF_8RowMajorENS11_17MmaTensorOpPolicyINSQ_3MmaINS7_ILi16ELi8ELi16EEELi32ESC_S14_SC_NSF_11ColumnMajorEfS14_NSQ_13OpMultiplyAddEEENSA_ILi1ELi1EEEEELi1ELb0EbEENSA_ILi0ELi0EEES1E_Li1EEELi10EbEENS_8epilogue11threadblock8EpilogueIS8_S1D_Li1ENS1I_22PredicatedTileIteratorINS1I_26OutputTileOptimalThreadMapINS1I_15OutputTileShapeILi64ELi8ELi2ELi1ELi1EEENS1M_ILi1ELi4ELi1ELi1ELi4EEELi128ELi4ELi32EEEfEENS1H_4warp24FragmentIteratorTensorOpIS13_S17_fNS_5ArrayIfLi4ELb1EEES14_EENS1R_20TileIteratorTensorOpIS13_S17_fS14_EENS1I_18SharedLoadIteratorINS1P_18CompactedThreadMapEfLi16EEENS1H_6thread17LinearCombinationIfLi4EffLNS_15FloatRoundStyleE2EEENSA_ILi0ELi8EEEEENSZ_30GemmIdentityThreadblockSwizzleILi4EEELNS1_8OperatorE2ENS1_17Conv2dProblemSizeEEEEEvNT_6ParamsEh ÄÄ*Ä2)8ÇË@ÇËHÇËXb;gradient_tape/model_1/conv2d_28/Conv2D/Conv2DBackpropFilterh
¨
Àvoid cutlass_cudnn::Kernel<cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4> >(cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4>::Params)6* 2@?8Ç¿@Ç¿HÇ¿Xb;gradient_tape/model_1/conv2d_41/Conv2D/Conv2DBackpropFilterhu  ñB
˛
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2Ä†8Ç¿@Å†HÅ†b/model_1/batch_normalization_14/FusedBatchNormV3hu  »B
Ÿ

ü
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8ÇÄ@ÇÄHÇÄbAdam/gradients/AddN_6huZUÖB
¨
Àvoid cutlass_cudnn::Kernel<cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4> >(cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4>::Params)6* 2@?8Ç®@Ç®HÇ®Xb;gradient_tape/model_1/conv2d_44/Conv2D/Conv2DBackpropFilterhu  ñB
˛
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2Äê8Ç®@Å–HÅÿb/model_1/batch_normalization_17/FusedBatchNormV3hu  »B
™
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å¿
@Å¿
HÅ¿
bAdam/gradients/AddN_29huZUÖB
˘
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ç∞
@Ç∞
HÇ∞
Xb;gradient_tape/model_1/conv2d_46/Conv2D/Conv2DBackpropFilterhuZUÖB
û
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Å∞
@Å∞
HÅ∞
b3gradient_tape/model_1/conv2d_29/BiasAdd/BiasAddGradhuZUÖB
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2¿8Ç®
@Ç®
HÇ®
b0gradient_tape/conv2d_46/kernel/Regularizer/Mul_1huZUÖB
†
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_64x3_ntÊÄÄ ÄÄ*Ä28Å†
@Å†
HÅ†
Xb;gradient_tape/model_1/conv2d_39/Conv2D/Conv2DBackpropFilterh
†
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_64x3_ntÊÄÄ ÄÄ*Ä28Çò
@Çò
HÇò
Xb;gradient_tape/model_1/conv2d_36/Conv2D/Conv2DBackpropFilterh
û
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Åê
@Åê
HÅê
b3gradient_tape/model_1/conv2d_34/BiasAdd/BiasAddGradhuZUÖB
¨
Àvoid cutlass_cudnn::Kernel<cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4> >(cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4>::Params)6* 2@28ÇÄ
@ÇÄ
HÇÄ
Xb;gradient_tape/model_1/conv2d_38/Conv2D/Conv2DBackpropFilterhu  ñB
û
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8ÇÄ
@ÇÄ
HÇÄ
b3gradient_tape/model_1/conv2d_28/BiasAdd/BiasAddGradhuZUÖB
û
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8ÇÄ
@ÇÄ
HÇÄ
b3gradient_tape/model_1/conv2d_31/BiasAdd/BiasAddGradhuZUÖB
û
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Ç¯	@Ç¯	HÇ¯	b3gradient_tape/model_1/conv2d_27/BiasAdd/BiasAddGradhuZUÖB
è
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2ÄP8Å‡	@Ä†HÅ†bAgradient_tape/model_1/batch_normalization_16/FusedBatchNormGradV3hu  »B
Ÿ

ü
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ç–	@Ç–	HÇ–	bAdam/gradients/AddN_7huZUÖB
Ÿ

ü
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ç¿	@Ç¿	HÇ¿	bAdam/gradients/AddN_9huZUÖB
Ö
√void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*Ä2{8Å∞	@ÄÄHÅÿbmodel_1/concatenate_13/concathuZUÖB
Ö
√void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*Ä2{8Ç®	@Å¯HÅÿbmodel_1/concatenate_12/concathuZUÖB
≠
Nvoid cudnn::ops::scalePackedTensor_kernel<__half, float>(long, __half*, float)*Ä2ˇˇ8Ç®	@Ç®	HÇ®	b9gradient_tape/model_1/max_pooling2d_5/MaxPool/MaxPoolGradhu  »B
≠
Nvoid cudnn::ops::scalePackedTensor_kernel<__half, float>(long, __half*, float)*Ä2ˇˇ8Å®	@Å®	HÅ®	b9gradient_tape/model_1/max_pooling2d_4/MaxPool/MaxPoolGradhu  »B
π
htensorflow::functor::ReluGradHalfKernelVector(Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2Ä(8Çò	@Çò	HÇò	b,gradient_tape/model_1/activation_15/ReluGradhu  »B
÷
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Åò	@Åò	HÅò	Xbmodel_1/conv2d_46/Conv2DhuZUÖB
π
htensorflow::functor::ReluGradHalfKernelVector(Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2Ä(8Çê	@Çê	HÇê	b,gradient_tape/model_1/activation_16/ReluGradhu  »B
¯
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Åê	@Åê	HÅê	Xb:gradient_tape/model_1/conv2d_46/Conv2D/Conv2DBackpropInputhuZUÖB
ñ
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8Åà	@Åà	HÅà	b<cond_1/then/_10/cond_1/Adam/Adam/update_60/ResourceApplyAdamhuZUÖB
ñ
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8ÅË@ÅËHÅËb<cond_1/then/_10/cond_1/Adam/Adam/update_44/ResourceApplyAdamhuZUÖB
ñ
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8Å‡@Å‡HÅ‡b<cond_1/then/_10/cond_1/Adam/Adam/update_36/ResourceApplyAdamhuZUÖB
ˇ
ívoid cudnn::bn_bw_1C11_singleread_specialized<__half2, 512, 1, 2, 7>(float, float, float, float, cudnnTensorStruct, __half2 const*, cudnnTensorStruct, __half2 const*, cudnnTensorStruct, __half2*, float const*, float*, float*, float const*, float const*, float, cudnn::reduced_divisor, int, cudnn::reduced_divisor, cudnn::bnBwPersistentState*, int, float, float, float, int, float, cudnnStatus_t*, bool)8ê Ä *Ä2Ä8Å–@Å–HÅ–bAgradient_tape/model_1/batch_normalization_17/FusedBatchNormGradV3huZUÖB
◊
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2{8Å∞@Å∞HÅ∞b/gradient_tape/conv2d_46/kernel/Regularizer/TilehuZUÖB
π
htensorflow::functor::ReluGradHalfKernelVector(Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2Ä$8Å®@Å®HÅ®b,gradient_tape/model_1/activation_18/ReluGradhu  »B
Ö
√void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*Ä2{8Å®@ÄÄHÄÿbmodel_1/concatenate_15/concathuZUÖB
ó
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Å†@Å†HÅ†b)gradient_tape/model_1/concatenate_9/SlicehuZUÖB
‡
ùvoid convolve_common_engine_float_NHWC<__half, __half, 128, 5, 5, 3, 3, 3, true, false, false, false, false>(int, int, int, __half const*, __half const*, int, __half*, conv_kernel_common_params, unsigned long long, unsigned long, float, float, bool, __half const*, __half const*, bool)PÄ*2ÄÄ8Å†@Å†HÅ†Xbmodel_1/conv2d_28/Conv2Dhu  HB
ô
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Çò@ÇòHÇòb+gradient_tape/model_1/concatenate_9/Slice_1huZUÖB
ò
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Çà@ÇàHÇàb*gradient_tape/model_1/concatenate_11/SlicehuZUÖB
ö
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Åà@ÅàHÅàb,gradient_tape/model_1/concatenate_11/Slice_2huZUÖB
ò
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8ÇÄ@ÇÄHÇÄb*gradient_tape/model_1/concatenate_10/SlicehuZUÖB
ö
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8ÅÄ@ÅÄHÅÄb,gradient_tape/model_1/concatenate_10/Slice_2huZUÖB
†
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_tn‚ÄÄ ÄÄ*Ä2Ä8ÇË@ÇËHÇËXb:gradient_tape/model_1/conv2d_39/Conv2D/Conv2DBackpropInputh
p
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä 8Ç∏@Ç∏HÇ∏b-gradient_tape/model_1/dropout_6/dropout/Mul_1huZUÖB
b
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä 8Ç∏@Ç∏HÇ∏bmodel_1/dropout_7/dropout/Mul_1huZUÖB
p
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä 8Å∏@Å∏HÅ∏b-gradient_tape/model_1/dropout_7/dropout/Mul_1huZUÖB
b
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä 8Å∞@Å∞HÅ∞bmodel_1/dropout_6/dropout/Mul_1huZUÖB
™
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å∞@Å∞HÅ∞bAdam/gradients/AddN_30huZUÖB
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2–8Å®@Å®HÅ®b0gradient_tape/conv2d_47/kernel/Regularizer/Mul_1huZUÖB
©
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Ç†@Ç†HÇ†bmodel_1/conv2d_32/BiasAddhuZUÖB
—
ıvoid cudnn::bn_fw_tr_1C11_singleread_specialized<__half2, 512, 1, 2, 20>(cudnnTensorStruct, __half2 const*, cudnnTensorStruct, __half2*, float const*, float const*, float, float, float*, float*, float*, float*, float, float, cudnn::reduced_divisor, int, cudnn::reduced_divisor, cudnn::bnFwPersistentState*, int, float, float, float, int, float, float, cudnnStatus_t*, bool)@ê Ä¿*Ä2Ä8Å†@Å†HÅ†b/model_1/batch_normalization_15/FusedBatchNormV3huZUÖB
©
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Åò@ÅòHÅòbmodel_1/conv2d_28/BiasAddhuZUÖB
©
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Åò@ÅòHÅòbmodel_1/conv2d_30/BiasAddhuZUÖB
©
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Åò@ÅòHÅòbmodel_1/conv2d_35/BiasAddhuZUÖB
©
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Åê@ÅêHÅêbmodel_1/conv2d_27/BiasAddhuZUÖB
©
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Åê@ÅêHÅêbmodel_1/conv2d_33/BiasAddhuZUÖB
K
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2¿8Åà@ÅàHÅàbmul_52huZUÖB
¶	
Ávoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Åà@ÅàHÅàbmodel_1/activation_16/ReluhuZUÖB
ñ
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8Åà@ÅàHÅàb<cond_1/then/_10/cond_1/Adam/Adam/update_28/ResourceApplyAdamhuZUÖB
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2¿8ÅÄ@ÅÄHÅÄb.gradient_tape/conv2d_46/kernel/Regularizer/MulhuZUÖB
k
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2¿8ÅÄ@ÅÄHÅÄb#conv2d_46/kernel/Regularizer/SquarehuZUÖB
÷
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Å–@Å–HÅ–Xbmodel_1/conv2d_47/Conv2DhuZUÖB
—
ıvoid cudnn::bn_fw_tr_1C11_singleread_specialized<__half2, 512, 1, 2, 20>(cudnnTensorStruct, __half2 const*, cudnnTensorStruct, __half2*, float const*, float const*, float, float, float*, float*, float*, float*, float, float, cudnn::reduced_divisor, int, cudnn::reduced_divisor, cudnn::bnFwPersistentState*, int, float, float, float, int, float, float, cudnnStatus_t*, bool)@ê Ä¿*Ä2Ä8Å»@Å»HÅ»b/model_1/batch_normalization_14/FusedBatchNormV3huZUÖB
¯
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Å»@Å»HÅ»Xb:gradient_tape/model_1/conv2d_47/Conv2D/Conv2DBackpropInputhuZUÖB
†
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_tn‚ÄÄ ÄÄ*Ä2Ä8Å¿@Å¿HÅ¿Xb:gradient_tape/model_1/conv2d_36/Conv2D/Conv2DBackpropInputh
˝
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2ÄP8Ç∞@ÅòHÅòb/model_1/batch_normalization_16/FusedBatchNormV3hu  »B
∏
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å∞@Å∞HÅ∞b0gradient_tape/model_1/conv2d_46/Conv2D/Cast/CasthuZUÖB
~
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_nnﬁÄÄ ÄÄ*Ä2Ä8Å®@Å®HÅ®Xbmodel_1/conv2d_39/Conv2Dh
}
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_nnﬁÄÄ ÄÄ*Ä2@8Å®@Å®HÅ®Xbmodel_1/conv2d_42/Conv2Dh
∞
Ìvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å®@Å®HÅ®bmodel_1/dropout_6/dropout/CasthuZUÖB
∞
Ìvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å®@Å®HÅ®bmodel_1/dropout_7/dropout/CasthuZUÖB
¶	
Ávoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å®@Å®HÅ®bmodel_1/activation_15/ReluhuZUÖB
ñ
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8Å®@Å®HÅ®b<cond_1/then/_10/cond_1/Adam/Adam/update_42/ResourceApplyAdamhuZUÖB
–
Ùvoid cudnn::bn_fw_tr_1C11_singleread_specialized<__half2, 512, 1, 2, 0>(cudnnTensorStruct, __half2 const*, cudnnTensorStruct, __half2*, float const*, float const*, float, float, float*, float*, float*, float*, float, float, cudnn::reduced_divisor, int, cudnn::reduced_divisor, cudnn::bnFwPersistentState*, int, float, float, float, int, float, float, cudnnStatus_t*, bool)(ê ÄÄ*Ä2Ä8Åò@ÅòHÅòb/model_1/batch_normalization_17/FusedBatchNormV3hu  »B
†
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_ntﬁÄÄ ÄÄ*Ä28Åê@ÅêHÅêXb;gradient_tape/model_1/conv2d_42/Conv2D/Conv2DBackpropFilterh
Ÿ

ü
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Åà@ÅàHÅàbAdam/gradients/AddN_4huZUÖB
±
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å¯@Å¯HÅ¯bmodel_1/conv2d_46/Conv2D/CasthuZUÖB
◊
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2{8ÅË@ÅËHÅËb/gradient_tape/conv2d_47/kernel/Regularizer/TilehuZUÖB
ñ
1ampere_s16816gemm_fp16_128x64_ldg8_stages_64x4_ntíÄÄ ÄÄ*Ä28Å‡@Å‡HÅ‡Xb;gradient_tape/model_1/conv2d_30/Conv2D/Conv2DBackpropFilterh
¶	
Ávoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Åÿ@ÅÿHÅÿbmodel_1/activation_18/ReluhuZUÖB
r
.ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_nnˆÄÄ*Ä2Ä8Å∏@Å∏HÅ∏Xbmodel_1/conv2d_30/Conv2DhugUÖA
ü
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_tn‚ÄÄ ÄÄ*Ä2@8Å∞@Å∞HÅ∞Xb:gradient_tape/model_1/conv2d_42/Conv2D/Conv2DBackpropInputh
î
.ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_tn˛ÄÄ*Ä2Ä8Å†@Å†HÅ†Xb:gradient_tape/model_1/conv2d_30/Conv2D/Conv2DBackpropInputhugUÖA
û
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Å†@Å†HÅ†b3gradient_tape/model_1/conv2d_32/BiasAdd/BiasAddGradhuZUÖB
û
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Å†@Å†HÅ†b3gradient_tape/model_1/conv2d_35/BiasAdd/BiasAddGradhuZUÖB
~
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_nnﬁÄÄ ÄÄ*Ä2Ä8Ä†@Ä†HÄ†Xbmodel_1/conv2d_36/Conv2Dh
û
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Åò@ÅòHÅòb3gradient_tape/model_1/conv2d_30/BiasAdd/BiasAddGradhuZUÖB
û
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Åò@ÅòHÅòb3gradient_tape/model_1/conv2d_33/BiasAdd/BiasAddGradhuZUÖB
û
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Åê@ÅêHÅêb3gradient_tape/model_1/conv2d_46/BiasAdd/BiasAddGradhuZUÖB
n
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä 8Äê@ÄêHÄêb+gradient_tape/model_1/dropout_7/dropout/MulhuZUÖB
n
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä 8Åà@ÅàHÅàb+gradient_tape/model_1/dropout_6/dropout/MulhuZUÖB
û
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  28Åà@ÅàHÅàb3gradient_tape/model_1/conv2d_43/BiasAdd/BiasAddGradhuZUÖB
û
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Åà@ÅàHÅàb3gradient_tape/model_1/conv2d_45/BiasAdd/BiasAddGradhuZUÖB
`
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä 8Äà@ÄàHÄàbmodel_1/dropout_7/dropout/MulhuZUÖB
`
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä 8ÅÄ@ÅÄHÅÄbmodel_1/dropout_6/dropout/MulhuZUÖB
û
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  28Å¯@Å¯HÅ¯b3gradient_tape/model_1/conv2d_42/BiasAdd/BiasAddGradhuZUÖB
K
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2–8Å@ÅHÅbmul_54huZUÖB
Ö
√void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*Ä2{8Å@ÄÄHÅ¯bmodel_1/concatenate_14/concathuZUÖB
ˇ
ívoid cudnn::bn_bw_1C11_singleread_specialized<__half2, 512, 1, 2, 7>(float, float, float, float, cudnnTensorStruct, __half2 const*, cudnnTensorStruct, __half2 const*, cudnnTensorStruct, __half2*, float const*, float*, float*, float const*, float const*, float, cudnn::reduced_divisor, int, cudnn::reduced_divisor, cudnn::bnBwPersistentState*, int, float, float, float, int, float, cudnnStatus_t*, bool)8ê Ä *Ä2Ä
8Å@ÅHÅbAgradient_tape/model_1/batch_normalization_16/FusedBatchNormGradV3huZUÖB
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2–8Ä@ÄHÄb.gradient_tape/conv2d_47/kernel/Regularizer/MulhuZUÖB
k
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2–8ÅË@ÅËHÅËb#conv2d_47/kernel/Regularizer/SquarehuZUÖB
˘
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8ÅË@ÅËHÅËXb;gradient_tape/model_1/conv2d_47/Conv2D/Conv2DBackpropFilterhuZUÖB
π
htensorflow::functor::ReluGradHalfKernelVector(Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2Ä8Å–@Å–HÅ–b,gradient_tape/model_1/activation_17/ReluGradhu  »B
¨
Àvoid cutlass_cudnn::Kernel<cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4> >(cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4>::Params)6* 2@8Ä»@Ä»HÄ»Xb;gradient_tape/model_1/conv2d_40/Conv2D/Conv2DBackpropFilterhu  ñB
“
åvoid pooling_fw_kernel_max_nhwc<__half, float, 0, (cudnnNanPropagation_t)0, false>(PoolingParams, __half*, __half const*, float, float, int)*Ä2Ä†8Å∞@Å∞HÅ∞bmodel_1/max_pooling2d_6/MaxPoolhu  »B
∏
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä∞@Ä∞HÄ∞b0gradient_tape/model_1/conv2d_47/Conv2D/Cast/CasthuZUÖB
T
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2ÄZ8Å®@Å®HÅ®bIsFinite_50hu  »B
±
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Äò@ÄòHÄòbmodel_1/conv2d_47/Conv2D/CasthuZUÖB
ö
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Äê@ÄêHÄêb,gradient_tape/model_1/concatenate_15/Slice_1huZUÖB
ö
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Åà@ÅàHÅàb,gradient_tape/model_1/concatenate_12/Slice_1huZUÖB
ö
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Åà@ÅàHÅàb,gradient_tape/model_1/concatenate_13/Slice_1huZUÖB
ò
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Åà@ÅàHÅàb*gradient_tape/model_1/concatenate_15/SlicehuZUÖB
“
åvoid pooling_fw_kernel_max_nhwc<__half, float, 0, (cudnnNanPropagation_t)0, false>(PoolingParams, __half*, __half const*, float, float, int)*Ä2Äê8Åà@ÅàHÅàbmodel_1/max_pooling2d_7/MaxPoolhu  »B
ö
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Äà@ÄàHÄàb,gradient_tape/model_1/concatenate_13/Slice_2huZUÖB
ö
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8ÅÄ@ÅÄHÅÄb,gradient_tape/model_1/concatenate_12/Slice_2huZUÖB
Ç
ßvoid tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>)*Ä2{8Å¯@Å¯HÅ¯b6model_1/dropout_7/dropout/random_uniform/RandomUniformhuZUÖB
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ä8ÅË@ÅËHÅËb.gradient_tape/dense_4/kernel/Regularizer/Mul_1huZUÖB
™
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8ÅË@ÅËHÅËbAdam/gradients/AddN_32huZUÖB
Ç
ßvoid tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>)*Ä2{8ÅË@ÅËHÅËb6model_1/dropout_6/dropout/random_uniform/RandomUniformhuZUÖB
¨
Àvoid cutlass_cudnn::Kernel<cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4> >(cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4>::Params)6* 2@8ÄË@ÄËHÄËXb;gradient_tape/model_1/conv2d_37/Conv2D/Conv2DBackpropFilterhu  ñB
r
'GreaterEqual_GPU_DT_HALF_DT_BOOL_kernel*Ä2Ä 8Å‡@Å‡HÅ‡b&model_1/dropout_7/dropout/GreaterEqualhuZUÖB
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ë8Å‡@Å‡HÅ‡b0gradient_tape/conv2d_44/kernel/Regularizer/Mul_1huZUÖB
™
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å‡@Å‡HÅ‡bAdam/gradients/AddN_27huZUÖB
©
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Å‡@Å‡HÅ‡bmodel_1/conv2d_38/BiasAddhuZUÖB
©
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Å‡@Å‡HÅ‡bmodel_1/conv2d_41/BiasAddhuZUÖB
©
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Å‡@Å‡HÅ‡bmodel_1/conv2d_46/BiasAddhuZUÖB
r
'GreaterEqual_GPU_DT_HALF_DT_BOOL_kernel*Ä2Ä 8Ä‡@Ä‡HÄ‡b&model_1/dropout_6/dropout/GreaterEqualhuZUÖB
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ë8Ä‡@Ä‡HÄ‡b0gradient_tape/conv2d_41/kernel/Regularizer/Mul_1huZUÖB
©
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Ä‡@Ä‡HÄ‡bmodel_1/conv2d_37/BiasAddhuZUÖB
©
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Ä‡@Ä‡HÄ‡bmodel_1/conv2d_45/BiasAddhuZUÖB
©
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Åÿ@ÅÿHÅÿbmodel_1/conv2d_40/BiasAddhuZUÖB
™
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Äÿ@ÄÿHÄÿbAdam/gradients/AddN_24huZUÖB
–
Ùvoid cudnn::bn_fw_tr_1C11_singleread_specialized<__half2, 512, 1, 2, 0>(cudnnTensorStruct, __half2 const*, cudnnTensorStruct, __half2*, float const*, float const*, float, float, float*, float*, float*, float*, float, float, cudnn::reduced_divisor, int, cudnn::reduced_divisor, cudnn::bnFwPersistentState*, int, float, float, float, int, float, float, cudnnStatus_t*, bool)(ê ÄÄ*Ä2Ä
8Äÿ@ÄÿHÄÿb/model_1/batch_normalization_16/FusedBatchNormV3hu  »B
ñ
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8Å–@Å–HÅ–b<cond_1/then/_10/cond_1/Adam/Adam/update_20/ResourceApplyAdamhuZUÖB
c
 ampere_fp16_sgemm_fp16_32x128_nn9ÄÄ*Ä2Ä 8Ä∏@Ä∏HÄ∏Xbmodel_1/conv2d_27/Conv2DhuZUÖB
÷
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Å∞@Å∞HÅ∞Xbmodel_1/conv2d_41/Conv2DhuZUÖB
√
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2ú8Ä∞@Ä∞HÄ∞b conv2d_46/kernel/Regularizer/Sumhu  »B
ñ
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8Å®@Å®HÅ®b<cond_1/then/_10/cond_1/Adam/Adam/update_34/ResourceApplyAdamhuZUÖB
¯
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Å®@Å®HÅ®Xb:gradient_tape/model_1/conv2d_41/Conv2D/Conv2DBackpropInputhuZUÖB
÷
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Å®@Å®HÅ®Xbmodel_1/conv2d_44/Conv2DhuZUÖB
¯
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ä®@Ä®HÄ®Xb:gradient_tape/model_1/conv2d_44/Conv2D/Conv2DBackpropInputhuZUÖB
å
§void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nt_align1>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nt_align1::Params)¢ Ä¿*Ä2¢8Å†@Å†HÅ†Xb;gradient_tape/model_1/conv2d_27/Conv2D/Conv2DBackpropFilterhugUÖA
¶	
Ávoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä†@Ä†HÄ†bmodel_1/activation_17/ReluhuZUÖB
Ÿ

ü
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Åò@ÅòHÅòbAdam/gradients/AddN_5huZUÖB
≠
Nvoid cudnn::ops::scalePackedTensor_kernel<__half, float>(long, __half*, float)*Ä2ˇˇ8Åê@ÅêHÅêb9gradient_tape/model_1/max_pooling2d_6/MaxPool/MaxPoolGradhu  »B
T
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2¿>8Åà@ÅàHÅàbIsFinite_52hu  »B
™
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Åà@ÅàHÅàbAdam/gradients/AddN_21huZUÖB
◊
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2{8Åà@ÅàHÅàb/gradient_tape/conv2d_41/kernel/Regularizer/TilehuZUÖB
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8ÅÄ@ÅÄHÅÄb0gradient_tape/conv2d_38/kernel/Regularizer/Mul_1huZUÖB
ñ
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8Ä¯@Ä¯HÄ¯b<cond_1/then/_10/cond_1/Adam/Adam/update_48/ResourceApplyAdamhuZUÖB
≠
Nvoid cudnn::ops::scalePackedTensor_kernel<__half, float>(long, __half*, float)*Ä2ˇˇ8Ä@ÄHÄb9gradient_tape/model_1/max_pooling2d_7/MaxPool/MaxPoolGradhu  »B
™
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8ÅË@ÅËHÅËbAdam/gradients/AddN_26huZUÖB
û
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8ÅË@ÅËHÅËb3gradient_tape/model_1/conv2d_37/BiasAdd/BiasAddGradhuZUÖB
û
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8ÅË@ÅËHÅËb3gradient_tape/model_1/conv2d_38/BiasAdd/BiasAddGradhuZUÖB
û
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8ÄË@ÄËHÄËb3gradient_tape/model_1/conv2d_40/BiasAdd/BiasAddGradhuZUÖB
û
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8ÄË@ÄËHÄËb3gradient_tape/model_1/conv2d_41/BiasAdd/BiasAddGradhuZUÖB
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2–8Åÿ@ÅÿHÅÿb0gradient_tape/conv2d_43/kernel/Regularizer/Mul_1huZUÖB
ñ
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8Åÿ@ÅÿHÅÿb<cond_1/then/_10/cond_1/Adam/Adam/update_26/ResourceApplyAdamhuZUÖB
ñ
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8Äÿ@ÄÿHÄÿb<cond_1/then/_10/cond_1/Adam/Adam/update_18/ResourceApplyAdamhuZUÖB
¯
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Äÿ@ÄÿHÄÿXb:gradient_tape/model_1/conv2d_38/Conv2D/Conv2DBackpropInputhuZUÖB
÷
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Å–@Å–HÅ–Xbmodel_1/conv2d_38/Conv2DhuZUÖB
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ë8Ä»@Ä»HÄ»b.gradient_tape/conv2d_41/kernel/Regularizer/MulhuZUÖB
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ë8Ä»@Ä»HÄ»b.gradient_tape/conv2d_44/kernel/Regularizer/MulhuZUÖB
q
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ä8Ä»@Ä»HÄ»b,gradient_tape/dense_4/kernel/Regularizer/MulhuZUÖB
K
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ä8Ä»@Ä»HÄ»bmul_62huZUÖB
i
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ä8Ä»@Ä»HÄ»b!dense_4/kernel/Regularizer/SquarehuZUÖB
K
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ë8Å¿@Å¿HÅ¿bmul_38huZUÖB
K
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ë8Å¿@Å¿HÅ¿bmul_46huZUÖB
k
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ë8Å¿@Å¿HÅ¿b#conv2d_41/kernel/Regularizer/SquarehuZUÖB
k
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ë8Å¿@Å¿HÅ¿b#conv2d_44/kernel/Regularizer/SquarehuZUÖB
◊
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2{8Å¿@Å¿HÅ¿b/gradient_tape/conv2d_38/kernel/Regularizer/TilehuZUÖB
√
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2–8Å¿@Å¿HÅ¿b conv2d_47/kernel/Regularizer/Sumhu  »B
p
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä
8Ä¿@Ä¿HÄ¿b-gradient_tape/model_1/dropout_8/dropout/Mul_1huZUÖB
÷
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ä¿@Ä¿HÄ¿Xbmodel_1/conv2d_43/Conv2DhuZUÖB
˘
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ä¿@Ä¿HÄ¿Xb;gradient_tape/model_1/conv2d_44/Conv2D/Conv2DBackpropFilterhuZUÖB
˘
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Å∏@Å∏HÅ∏Xb;gradient_tape/model_1/conv2d_41/Conv2D/Conv2DBackpropFilterhuZUÖB
b
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä
8Ä∏@Ä∏HÄ∏bmodel_1/dropout_8/dropout/Mul_1huZUÖB
¯
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ä∏@Ä∏HÄ∏Xb:gradient_tape/model_1/conv2d_43/Conv2D/Conv2DBackpropInputhuZUÖB
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä∞@Ä∞HÄ∞b.gradient_tape/model_1/dense_4/MatMul/Cast/CasthuZUÖB
◊
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2{8Ä®@Ä®HÄ®b/gradient_tape/conv2d_43/kernel/Regularizer/TilehuZUÖB
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å†@Å†HÅ†bmodel_1/dense_4/MatMul/CasthuZUÖB
∏
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å†@Å†HÅ†b0gradient_tape/model_1/conv2d_41/Conv2D/Cast/CasthuZUÖB
p
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä	8Ä†@Ä†HÄ†b-gradient_tape/model_1/dropout_9/dropout/Mul_1huZUÖB
∏
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä†@Ä†HÄ†b0gradient_tape/model_1/conv2d_44/Conv2D/Cast/CasthuZUÖB
b
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä	8Äò@ÄòHÄòbmodel_1/dropout_9/dropout/Mul_1huZUÖB
±
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Äò@ÄòHÄòbmodel_1/conv2d_41/Conv2D/CasthuZUÖB
±
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Äò@ÄòHÄòbmodel_1/conv2d_44/Conv2D/CasthuZUÖB
∞
Ìvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Åê@ÅêHÅêbmodel_1/dropout_8/dropout/CasthuZUÖB
ö
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Åê@ÅêHÅêb,gradient_tape/model_1/concatenate_14/Slice_1huZUÖB
K
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8Äê@ÄêHÄêbmul_30huZUÖB
ò
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Äê@ÄêHÄêb*gradient_tape/model_1/concatenate_12/SlicehuZUÖB
ò
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Äê@ÄêHÄêb*gradient_tape/model_1/concatenate_13/SlicehuZUÖB
ò
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Äê@ÄêHÄêb*gradient_tape/model_1/concatenate_14/SlicehuZUÖB
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8Äà@ÄàHÄàb.gradient_tape/conv2d_38/kernel/Regularizer/MulhuZUÖB
©
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8ÅÄ@ÅÄHÅÄbmodel_1/conv2d_42/BiasAddhuZUÖB
k
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8ÄÄ@ÄÄHÄÄb#conv2d_38/kernel/Regularizer/SquarehuZUÖB
©
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8ÄÄ@ÄÄHÄÄbmodel_1/conv2d_39/BiasAddhuZUÖB
˘
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8ÄÄ@ÄÄHÄÄXb;gradient_tape/model_1/conv2d_38/Conv2D/Conv2DBackpropFilterhuZUÖB
∞
Ìvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å¯@Å¯HÅ¯bmodel_1/dropout_9/dropout/CasthuZUÖB
©
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Ä¯@Ä¯HÄ¯bmodel_1/conv2d_36/BiasAddhuZUÖB
©
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Ä¯@Ä¯HÄ¯bmodel_1/conv2d_43/BiasAddhuZUÖB
˘
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Å@ÅHÅXb;gradient_tape/model_1/conv2d_43/Conv2D/Conv2DBackpropFilterhuZUÖB
K
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2–8Ä@ÄHÄbmul_44huZUÖB
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2–8ÅË@ÅËHÅËb.gradient_tape/conv2d_43/kernel/Regularizer/MulhuZUÖB
k
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2–8ÄË@ÄËHÄËb#conv2d_43/kernel/Regularizer/SquarehuZUÖB
±
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8ÄË@ÄËHÄËbmodel_1/conv2d_38/Conv2D/CasthuZUÖB
∏
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8ÄË@ÄËHÄËb0gradient_tape/model_1/conv2d_38/Conv2D/Cast/CasthuZUÖB
±
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Åÿ@ÅÿHÅÿbmodel_1/conv2d_43/Conv2D/CasthuZUÖB
T
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2Ä 8Äÿ@ÄÿHÄÿbIsFinite_60hu  »B
n
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä	8Äÿ@ÄÿHÄÿb+gradient_tape/model_1/dropout_9/dropout/MulhuZUÖB
n
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä
8Äÿ@ÄÿHÄÿb+gradient_tape/model_1/dropout_8/dropout/MulhuZUÖB
T
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2†8Ä–@Ä–HÄ–bIsFinite_36hu  »B
∏
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä–@Ä–HÄ–b0gradient_tape/model_1/conv2d_43/Conv2D/Cast/CasthuZUÖB
™
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä–@Ä–HÄ–bAdam/gradients/AddN_18huZUÖB
T
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2†8Ä»@Ä»HÄ»bIsFinite_44hu  »B
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2ê8Ä»@Ä»HÄ»b0gradient_tape/conv2d_35/kernel/Regularizer/Mul_1huZUÖB
`
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä
8Ä»@Ä»HÄ»bmodel_1/dropout_8/dropout/MulhuZUÖB
’
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Ä»@Ä»HÄ»b-gradient_tape/dense_4/kernel/Regularizer/TilehuZUÖB
û
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Å¿@Å¿HÅ¿b3gradient_tape/model_1/conv2d_39/BiasAdd/BiasAddGradhuZUÖB
`
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä	8Ä¿@Ä¿HÄ¿bmodel_1/dropout_9/dropout/MulhuZUÖB
™
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä∏@Ä∏HÄ∏bAdam/gradients/AddN_23huZUÖB
û
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Ä∏@Ä∏HÄ∏b3gradient_tape/model_1/conv2d_36/BiasAdd/BiasAddGradhuZUÖB
¯
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ä∏@Ä∏HÄ∏Xb:gradient_tape/model_1/conv2d_35/Conv2D/Conv2DBackpropInputhuZUÖB
÷
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ä∏@Ä∏HÄ∏Xbmodel_1/conv2d_35/Conv2DhuZUÖB
√
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2Ë8Å∞@Å∞HÅ∞b conv2d_44/kernel/Regularizer/Sumhu  »B
÷
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Å∞@Å∞HÅ∞Xbmodel_1/conv2d_40/Conv2DhuZUÖB
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ë8Ä∞@Ä∞HÄ∞b0gradient_tape/conv2d_40/kernel/Regularizer/Mul_1huZUÖB
É
Campere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x5_nníÄÄ ÄÄ*Ä2 8Ä∞@Ä∞HÄ∞Xbmodel_1/dense_4/MatMulh
ë
Campere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tníÄÄ ÄÄ*Ä2 8Ä∞@Ä∞HÄ∞Xb$gradient_tape/model_1/dense_4/MatMulh
Ç
ßvoid tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>)*Ä2{8Ä∞@Ä∞HÄ∞b6model_1/dropout_8/dropout/random_uniform/RandomUniformhuZUÖB
™
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å®@Å®HÅ®bAdam/gradients/AddN_28huZUÖB
r
'GreaterEqual_GPU_DT_HALF_DT_BOOL_kernel*Ä2Ä
8Ä®@Ä®HÄ®b&model_1/dropout_8/dropout/GreaterEqualhuZUÖB
T
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2Ä8Ä®@Ä®HÄ®bIsFinite_28hu  »B
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2¿8Ä®@Ä®HÄ®b0gradient_tape/conv2d_45/kernel/Regularizer/Mul_1huZUÖB
√
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2Ë8Ä®@Ä®HÄ®b conv2d_41/kernel/Regularizer/Sumhu  »B
¡
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2Ä8Ä®@Ä®HÄ®bdense_4/kernel/Regularizer/Sumhu  »B
‹
˙void nhwcAddPaddingKernel<__half, __half, float, true, (cudnnKernelDataType_t)0>(int, int, int, int, int, int, int, int, __half const*, __half*, int, int, int, int, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor)*Ä2§8Ä®@ÄHÄêXb;gradient_tape/model_1/conv2d_29/Conv2D/Conv2DBackpropFilterhu  »B
¯
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ä®@Ä®HÄ®Xb:gradient_tape/model_1/conv2d_40/Conv2D/Conv2DBackpropInputhuZUÖB
T
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2¿8Ä†@Ä†HÄ†bIsFinite_42hu  »B
◊
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2{8Ä†@Ä†HÄ†b/gradient_tape/conv2d_35/kernel/Regularizer/TilehuZUÖB
◊
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2{8Ä†@Ä†HÄ†b/gradient_tape/conv2d_40/kernel/Regularizer/TilehuZUÖB
√
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2†8Ä†@Ä†HÄ†b conv2d_38/kernel/Regularizer/Sumhu  »B
‹
˙void nhwcAddPaddingKernel<__half, __half, float, true, (cudnnKernelDataType_t)0>(int, int, int, int, int, int, int, int, __half const*, __half*, int, int, int, int, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor)*Ä2§8Ä†@ÄHÄêXb;gradient_tape/model_1/conv2d_28/Conv2D/Conv2DBackpropFilterhu  »B
Ç
ßvoid tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>)*Ä2{8Ä†@Ä†HÄ†b6model_1/dropout_9/dropout/random_uniform/RandomUniformhuZUÖB
ö
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Åò@ÅòHÅòb,gradient_tape/model_1/concatenate_15/Slice_2huZUÖB
™
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Åò@ÅòHÅòbAdam/gradients/AddN_20huZUÖB
r
'GreaterEqual_GPU_DT_HALF_DT_BOOL_kernel*Ä2Ä	8Äò@ÄòHÄòb&model_1/dropout_9/dropout/GreaterEqualhuZUÖB
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8Äò@ÄòHÄòb0gradient_tape/conv2d_34/kernel/Regularizer/Mul_1huZUÖB
™
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Äò@ÄòHÄòbAdam/gradients/AddN_17huZUÖB
√
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2–8Äò@ÄòHÄòb conv2d_43/kernel/Regularizer/Sumhu  »B
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8Äê@ÄêHÄêb0gradient_tape/conv2d_37/kernel/Regularizer/Mul_1huZUÖB
K
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2ê8Äê@ÄêHÄêbmul_22huZUÖB
ö
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Äê@ÄêHÄêb,gradient_tape/model_1/concatenate_14/Slice_2huZUÖB
å
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2†8Äê@ÄêHÄêbAll_50hu  »B
©
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Äê@ÄêHÄêbmodel_1/conv2d_44/BiasAddhuZUÖB
©
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Äê@ÄêHÄêbmodel_1/conv2d_47/BiasAddhuZUÖB
¯
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Äê@ÄêHÄêXb:gradient_tape/model_1/conv2d_34/Conv2D/Conv2DBackpropInputhuZUÖB
¯
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Äê@ÄêHÄêXb:gradient_tape/model_1/conv2d_37/Conv2D/Conv2DBackpropInputhuZUÖB
÷
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Äê@ÄêHÄêXbmodel_1/conv2d_34/Conv2DhuZUÖB
˘
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Äê@ÄêHÄêXb;gradient_tape/model_1/conv2d_35/Conv2D/Conv2DBackpropFilterhuZUÖB
◊
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2{8Åà@ÅàHÅàb/gradient_tape/conv2d_34/kernel/Regularizer/TilehuZUÖB
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2ê8Äà@ÄàHÄàb.gradient_tape/conv2d_35/kernel/Regularizer/MulhuZUÖB
k
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2ê8Äà@ÄàHÄàb#conv2d_35/kernel/Regularizer/SquarehuZUÖB
◊
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2{8Äà@ÄàHÄàb/gradient_tape/conv2d_45/kernel/Regularizer/TilehuZUÖB
ñ
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8Äà@ÄàHÄàb<cond_1/then/_10/cond_1/Adam/Adam/update_12/ResourceApplyAdamhuZUÖB
÷
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Äà@ÄàHÄàXbmodel_1/conv2d_37/Conv2DhuZUÖB
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ë8ÄÄ@ÄÄHÄÄb.gradient_tape/conv2d_40/kernel/Regularizer/MulhuZUÖB
K
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ë8ÄÄ@ÄÄHÄÄbmul_36huZUÖB
k
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ë8ÄÄ@ÄÄHÄÄb#conv2d_40/kernel/Regularizer/SquarehuZUÖB
±
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8ÄÄ@ÄÄHÄÄbmodel_1/conv2d_35/Conv2D/CasthuZUÖB
∏
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8ÄÄ@ÄÄHÄÄb0gradient_tape/model_1/conv2d_35/Conv2D/Cast/CasthuZUÖB
H
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2¿8Åx@ÅxHÅxbmul_50huZUÖB
ˆ
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Åx@ÅxHÅxXb;gradient_tape/model_1/conv2d_40/Conv2D/Conv2DBackpropFilterhuZUÖB
ß
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Äx@ÄxHÄxbmodel_1/conv2d_27/CasthuZUÖB
Æ
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Äx@ÄxHÄxbmodel_1/conv2d_40/Conv2D/CasthuZUÖB
ï
Ávoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<unsigned char const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<unsigned char const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Äx@ÄxHÄxbmodel_1/CasthuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8Äx@ÄxHÄxb<cond_1/then/_10/cond_1/Adam/Adam/update_40/ResourceApplyAdamhuZUÖB
õ
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Äx@ÄxHÄxb3gradient_tape/model_1/conv2d_44/BiasAdd/BiasAddGradhuZUÖB
õ
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Äx@ÄxHÄxb3gradient_tape/model_1/conv2d_47/BiasAdd/BiasAddGradhuZUÖB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8Äp@ÄpHÄpb.gradient_tape/conv2d_37/kernel/Regularizer/MulhuZUÖB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2¿8Äp@ÄpHÄpb.gradient_tape/conv2d_45/kernel/Regularizer/MulhuZUÖB
Ì
ëvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorForcedEvalOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorForcedEvalOp<Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_log_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorForcedEvalOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorForcedEvalOp<Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_log_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long)2*Ä28Äp@ÄpHÄpb:categorical_crossentropy/softmax_cross_entropy_with_logitshuZUÖB
Ü
:ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_ntÍÄÄ*Ä28Åh@ÅhHÅhb&gradient_tape/model_1/dense_4/MatMul_1hugUÖA
â
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2Ë8Åh@ÅhHÅhbAll_52hu  »B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8Äh@ÄhHÄhb.gradient_tape/conv2d_34/kernel/Regularizer/MulhuZUÖB
H
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8Äh@ÄhHÄhbmul_20huZUÖB
H
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8Äh@ÄhHÄhbmul_28huZUÖB
h
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8Äh@ÄhHÄhb#conv2d_34/kernel/Regularizer/SquarehuZUÖB
h
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2¿8Äh@ÄhHÄhb#conv2d_45/kernel/Regularizer/SquarehuZUÖB
Æ
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Äh@ÄhHÄhbmodel_1/conv2d_45/Conv2D/CasthuZUÖB
À
Êvoid xmma_cudnn::gemm::split_k_kernel<xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, xmma_cudnn::Col, 256, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 3> >(xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, xmma_cudnn::Col, 256, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 3>::Params)? Äê*Ä2	8Äh@ÄhHÄhPXb;gradient_tape/model_1/conv2d_31/Conv2D/Conv2DBackpropFilterhuMUB
h
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8Å`@Å`HÅ`b#conv2d_37/kernel/Regularizer/SquarehuZUÖB
Æ
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä`@Ä`HÄ`bmodel_1/conv2d_34/Conv2D/CasthuZUÖB
Æ
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä`@Ä`HÄ`bmodel_1/conv2d_37/Conv2D/CasthuZUÖB
µ
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä`@Ä`HÄ`b0gradient_tape/model_1/conv2d_40/Conv2D/Cast/CasthuZUÖB
µ
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä`@Ä`HÄ`b0gradient_tape/model_1/conv2d_45/Conv2D/Cast/CasthuZUÖB
¿
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2ê8Ä`@Ä`HÄ`b conv2d_35/kernel/Regularizer/Sumhu  »B
ˆ
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ä`@Ä`HÄ`Xb;gradient_tape/model_1/conv2d_34/Conv2D/Conv2DBackpropFilterhuZUÖB
ˆ
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ä`@Ä`HÄ`Xb;gradient_tape/model_1/conv2d_37/Conv2D/Conv2DBackpropFilterhuZUÖB
Q
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2†8ÄX@ÄXHÄXbIsFinite_34hu  »B
Q
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2¿8ÄX@ÄXHÄXbIsFinite_20hu  »B
Q
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2Ä	8ÄP@ÄPHÄPbIsFinite_26hu  »B
Q
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2Ä
8ÄP@ÄPHÄPbIsFinite_48hu  »B
µ
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8ÄP@ÄPHÄPb0gradient_tape/model_1/conv2d_37/Conv2D/Cast/CasthuZUÖB
¿
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2Ë8ÄP@ÄPHÄPb conv2d_40/kernel/Regularizer/Sumhu  »B
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8ÄP@ÄPHÄPb<cond_1/then/_10/cond_1/Adam/Adam/update_10/ResourceApplyAdamhuZUÖB
Q
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2Ä	8ÄH@ÄHHÄHbIsFinite_18hu  »B
µ
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8ÄH@ÄHHÄHb0gradient_tape/model_1/conv2d_34/Conv2D/Cast/CasthuZUÖB
‘
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2{8ÄH@ÄHHÄHb/gradient_tape/conv2d_32/kernel/Regularizer/TilehuZUÖB
Î
èvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorReshapingOp<Eigen::IndexList<int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> >, Eigen::TensorReductionOp<Eigen::internal::MaxReducer<float, 0>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorReshapingOp<Eigen::IndexList<int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> >, Eigen::TensorReductionOp<Eigen::internal::MaxReducer<float, 0>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long)@*Ä28ÄH@ÄHHÄHb:categorical_crossentropy/softmax_cross_entropy_with_logitshuZUÖB
¿
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2¿8ÄH@ÄHHÄHb conv2d_45/kernel/Regularizer/Sumhu  »B
ı
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8ÄH@ÄHHÄHXb:gradient_tape/model_1/conv2d_32/Conv2D/Conv2DBackpropInputhuZUÖB
q
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2d8Ä@@Ä@HÄ@b0gradient_tape/conv2d_32/kernel/Regularizer/Mul_1huZUÖB
ß
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä@@Ä@HÄ@bAdam/gradients/AddN_15huZUÖB
â
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2ê8Ä@@Ä@HÄ@bAll_28hu  »B
â
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2Ù8Ä@@Ä@HÄ@bAll_44hu  »B
¿
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2†8Ä@@Ä@HÄ@b conv2d_34/kernel/Regularizer/Sumhu  »B
¿
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2†8Ä@@Ä@HÄ@b conv2d_37/kernel/Regularizer/Sumhu  »B
”
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ä@@Ä@HÄ@Xbmodel_1/conv2d_32/Conv2DhuZUÖB
”
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Å8@Å8HÅ8Xbmodel_1/conv2d_31/Conv2DhuZUÖB
q
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2H8Ä8@Ä8HÄ8b0gradient_tape/conv2d_31/kernel/Regularizer/Mul_1huZUÖB
q
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2P8Ä8@Ä8HÄ8b0gradient_tape/conv2d_42/kernel/Regularizer/Mul_1huZUÖB
ß
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä8@Ä8HÄ8bAdam/gradients/AddN_25huZUÖB
Â
âvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorReshapingOp<Eigen::IndexList<int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> >, Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorReshapingOp<Eigen::IndexList<int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> >, Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long)/*Ä28Ä8@Ä8HÄ8b:categorical_crossentropy/softmax_cross_entropy_with_logitshuZUÖB
â
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2Ë8Ä8@Ä8HÄ8bAll_42hu  »B
â
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2Ù8Ä8@Ä8HÄ8bAll_36hu  »B
â
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2Ä8Ä8@Ä8HÄ8bAll_60hu  »B
ı
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ä8@Ä8HÄ8Xb:gradient_tape/model_1/conv2d_31/Conv2D/Conv2DBackpropInputhuZUÖB
ˆ
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ä8@Ä8HÄ8Xb;gradient_tape/model_1/conv2d_32/Conv2D/Conv2DBackpropFilterhuZUÖB
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2H8Å0@Å0HÅ0b.gradient_tape/conv2d_31/kernel/Regularizer/MulhuZUÖB
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2P8Ä0@Ä0HÄ0b.gradient_tape/conv2d_42/kernel/Regularizer/MulhuZUÖB
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2d8Ä0@Ä0HÄ0b.gradient_tape/conv2d_32/kernel/Regularizer/MulhuZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2d8Ä0@Ä0HÄ0bmul_14huZUÖB
ß
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä0@Ä0HÄ0bAdam/gradients/AddN_14huZUÖB
‘
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2{8Ä0@Ä0HÄ0b/gradient_tape/conv2d_31/kernel/Regularizer/TilehuZUÖB
‘
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2{8Ä0@Ä0HÄ0b/gradient_tape/conv2d_42/kernel/Regularizer/TilehuZUÖB
â
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2†8Ä0@Ä0HÄ0bAll_48hu  »B
©
Àvoid cutlass_cudnn::Kernel<cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4> >(cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4>::Params)6* 28Ä0@Ä0HÄ0Xb;gradient_tape/model_1/conv2d_29/Conv2D/Conv2DBackpropFilterhu  ñB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2P8Ä0@Ä0HÄ0b<cond_1/then/_10/cond_1/Adam/Adam/update_32/ResourceApplyAdamhuZUÖB
ˆ
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ä0@Ä0HÄ0Xb;gradient_tape/model_1/conv2d_31/Conv2D/Conv2DBackpropFilterhuZUÖB
â
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Å(@Å(HÅ(b3gradient_tape/model_1/conv2d_34/BiasAdd/BiasAddGradhuZUÖB
Q
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2ê8Ä(@Ä(HÄ(bIsFinite_12hu  »B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2H8Ä(@Ä(HÄ(bmul_12huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2P8Ä(@Ä(HÄ(bmul_42huZUÖB
g
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2H8Ä(@Ä(HÄ(b#conv2d_31/kernel/Regularizer/SquarehuZUÖB
g
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2P8Ä(@Ä(HÄ(b#conv2d_42/kernel/Regularizer/SquarehuZUÖB
g
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2d8Ä(@Ä(HÄ(b#conv2d_32/kernel/Regularizer/SquarehuZUÖB
µ
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä(@Ä(HÄ(b0gradient_tape/model_1/conv2d_32/Conv2D/Cast/CasthuZUÖB
›
ìvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long) *Ä28Ä(@ÄHÄb(ArithmeticOptimizer/AddOpsRewrite_AddN_1huZUÖB
‘
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä28Ä(@Ä(HÄ(b/gradient_tape/conv2d_29/kernel/Regularizer/TilehuZUÖB
‘
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä28Ä(@Ä(HÄ(b/gradient_tape/conv2d_30/kernel/Regularizer/TilehuZUÖB
‘
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2@8Ä(@Ä(HÄ(b/gradient_tape/conv2d_33/kernel/Regularizer/TilehuZUÖB
‘
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2P8Ä(@Ä(HÄ(b/gradient_tape/conv2d_39/kernel/Regularizer/TilehuZUÖB
â
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2ê8Ä(@Ä(HÄ(bAll_18hu  »B
â
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2¥8Ä(@Ä(HÄ(bAll_34hu  »B
â
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2»8Ä(@Ä(HÄ(bAll_20hu  »B
ø
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2d8Ä(@Ä(HÄ(b conv2d_32/kernel/Regularizer/Sumhu  »B
©
Àvoid cutlass_cudnn::Kernel<cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4> >(cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4>::Params)6* 28Ä(@Ä(HÄ(Xb;gradient_tape/model_1/conv2d_28/Conv2D/Conv2DBackpropFilterhu  ñB
É
§void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*) *Ä28Ä(@Ä(HÄ(Xb;gradient_tape/model_1/conv2d_27/Conv2D/Conv2DBackpropFilterhu  »B
Ñ
§void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*) *Ä2Ä8Ä(@Ä(HÄ(Xb;gradient_tape/model_1/conv2d_42/Conv2D/Conv2DBackpropFilterhu  »B
Ë
©void tensorflow::(anonymous namespace)::GenerateNormalizedProb<Eigen::half, float, 8>(Eigen::half const*, float const*, Eigen::half const*, Eigen::half*, int, int, bool)*Ä28Ä(@Ä(HÄ(bmodel_1/activation_21/Softmaxhu  »B
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2@8Ä(@Ä(HÄ(b<cond_1/then/_10/cond_1/Adam/Adam/update_24/ResourceApplyAdamhuZUÖB
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä(@Ä(HÄ(bAll_54hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä(@Ä(HÄ(bAll_55hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä(@Ä(HÄ(bAll_61hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä2 8Ä(@Ä(HÄ(bAll_16hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä2 8Ä(@Ä(HÄ(bAll_24hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä2 8Ä(@Ä(HÄ(bAll_32hu  »B
â
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Ä(@Ä(HÄ(b3gradient_tape/model_1/conv2d_31/BiasAdd/BiasAddGradhuZUÖB
â
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Ä(@Ä(HÄ(b3gradient_tape/model_1/conv2d_37/BiasAdd/BiasAddGradhuZUÖB
â
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Ä(@Ä(HÄ(b3gradient_tape/model_1/conv2d_38/BiasAdd/BiasAddGradhuZUÖB
â
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Ä(@Ä(HÄ(b3gradient_tape/model_1/conv2d_40/BiasAdd/BiasAddGradhuZUÖB
â
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Ä(@Ä(HÄ(b3gradient_tape/model_1/conv2d_41/BiasAdd/BiasAddGradhuZUÖB
â
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Ä(@Ä(HÄ(b3gradient_tape/model_1/conv2d_44/BiasAdd/BiasAddGradhuZUÖB
â
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Ä(@Ä(HÄ(b3gradient_tape/model_1/conv2d_47/BiasAdd/BiasAddGradhuZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Å @Å HÅ bmul_34huZUÖB
≠
—void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorEvalToOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorForcedEvalOp<Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_log_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorEvalToOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorForcedEvalOp<Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_log_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>, long)(*Ä28Å @Å HÅ b:categorical_crossentropy/softmax_cross_entropy_with_logitshuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Å @Å HÅ b<cond_1/then/_10/cond_1/Adam/Adam/update_47/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Å @Å HÅ b<cond_1/then/_10/cond_1/Adam/Adam/update_61/ResourceApplyAdamhuZUÖB
â
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Å @Å HÅ b3gradient_tape/model_1/conv2d_36/BiasAdd/BiasAddGradhuZUÖB
Ä
ßvoid tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>)*Ä2@8Å @Å HÅ b7model_1/dropout_10/dropout/random_uniform/RandomUniformhuZUÖB
Q
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2†8Ä @Ä HÄ bIsFinite_10hu  »B
Q
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2¿8Ä @Ä HÄ bIsFinite_40hu  »B
k
 ampere_fp16_sgemm_fp16_128x32_tn9ÄÄ*Ä28Ä @Ä HÄ Xb$gradient_tape/model_1/dense_5/MatMulhuZUÖB
µ
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä @Ä HÄ b0gradient_tape/model_1/conv2d_31/Conv2D/Cast/CasthuZUÖB
µ
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä @Ä HÄ b0gradient_tape/model_1/conv2d_42/Conv2D/Cast/CasthuZUÖB
ß
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä @Ä HÄ bAdam/gradients/AddN_19huZUÖB
¡	
Ávoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_squared_difference_op<float>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_squared_difference_op<float>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, int) *Ä2@8Ä @Ä HÄ b8model_1/batch_normalization_19/moments/SquaredDifferencehuZUÖB
≈
Èvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_quotient_op<float, float>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_quotient_op<float, float>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long).*Ä28Ä @Ä HÄ b:categorical_crossentropy/softmax_cross_entropy_with_logitshuZUÖB
‘
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä28Ä @Ä HÄ b/gradient_tape/conv2d_27/kernel/Regularizer/TilehuZUÖB
â
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2ê8Ä @Ä HÄ bAll_26hu  »B
ø
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2H8Ä @Ä HÄ b conv2d_31/kernel/Regularizer/Sumhu  »B
ø
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2P8Ä @Ä HÄ b conv2d_42/kernel/Regularizer/Sumhu  »B
Ö
€void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *Ä28Ä @Ä HÄ bAll_56hu¶™¶B
ª
˘void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float>, float>(float*, float*, int, tensorflow::functor::Sum<float>, float)0*Ä28Ä @Ä HÄ bdense_3/kernel/Regularizer/Sumhu  »B
‚
§void cutlass::Kernel<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2>(cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2::Params)_ Ä§*Ä28Ä @Ä HÄ Xbmodel_1/dense_5/MatMulhugUÖA

§void cutlass::Kernel<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nt_align2>(cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nt_align2::Params)_ Ä¿*Ä2@8Ä @Ä HÄ b&gradient_tape/model_1/dense_5/MatMul_1hugUÖA
Ñ
§void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*) *Ä2Ä8Ä @Ä HÄ Xb;gradient_tape/model_1/conv2d_33/Conv2D/Conv2DBackpropFilterhu  »B
ﬂ
§void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*) *Ä2Ä8Ä @Ä HÄ Xbmodel_1/dense_3/MatMulhu  »B
Ñ
§void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*) *Ä2Ä8Ä @Ä HÄ Xb;gradient_tape/model_1/conv2d_39/Conv2D/Conv2DBackpropFilterhu  »B
ê
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b9cond_1/then/_10/cond_1/Adam/Adam/update/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_14/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_15/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_17/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_21/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_23/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_27/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_29/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_30/ResourceApplyAdamhuZUÖB
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
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_46/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_57/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_58/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_59/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_62/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_63/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_54/ResourceApplyAdamhuZUÖB
í
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b;cond_1/then/_10/cond_1/Adam/Adam/update_4/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_64/ResourceApplyAdamhuZUÖB
í
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b;cond_1/then/_10/cond_1/Adam/Adam/update_8/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2@8Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_16/ResourceApplyAdamhuZUÖB
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä @Ä HÄ bAll_46hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä @Ä HÄ bAll_49hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä @Ä HÄ bAll_51hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä @Ä HÄ bAll_57hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä @Ä HÄ bAll_59hu  »B
◊
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä2 8Ä @Ä HÄ bAll_8hu  »B
Ü
¬void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*Ä2 8Ä @Ä HÄ b conv2d_33/kernel/Regularizer/Sumhu  »B
â
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Ä @Ä HÄ b3gradient_tape/model_1/conv2d_30/BiasAdd/BiasAddGradhuZUÖB
â
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Ä @Ä HÄ b3gradient_tape/model_1/conv2d_33/BiasAdd/BiasAddGradhuZUÖB
â
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Ä @Ä HÄ b3gradient_tape/model_1/conv2d_39/BiasAdd/BiasAddGradhuZUÖB
Ä
ßvoid tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>)*Ä2@8Ä @Ä HÄ b7model_1/dropout_11/dropout/random_uniform/RandomUniformhuZUÖB
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Å@ÅHÅb conv2d_30/kernel/Regularizer/mulhuZUÖB
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2@8Å@ÅHÅb.model_1/batch_normalization_19/batchnorm/mul_1huZUÖB
Ω
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Å@ÅHÅb8gradient_tape/model_1/batch_normalization_18/Cast_1/CasthuZUÖB
ô
’void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Å@ÅHÅb"cond_1/then/_10/cond_1/Adam/Cast_1huZUÖB
Ω
˘void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float>, float>(float*, float*, int, tensorflow::functor::Sum<float>, float)0*Ä28Å@ÅHÅb conv2d_37/kernel/Regularizer/Sumhu  »B
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Å@ÅHÅb<cond_1/then/_10/cond_1/Adam/Adam/update_38/ResourceApplyAdamhuZUÖB
•
–void tensorflow::functor::ColumnReduceMax16ColumnsKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿* 28Å@ÅHÅb1gradient_tape/model_1/dense_5/BiasAdd/BiasAddGradhu  ØB
o
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb,model_1/batch_normalization_18/batchnorm/addhuZUÖB
o
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb,model_1/batch_normalization_19/batchnorm/addhuZUÖB
q
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2@8Ä@ÄHÄb.model_1/batch_normalization_18/batchnorm/add_1huZUÖB
q
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2@8Ä@ÄHÄb.model_1/batch_normalization_19/batchnorm/add_1huZUÖB
X
"AddV2_GPU_DT_INT64_DT_INT64_kernel*Ä28Ä@ÄHÄbcond/then/_0/cond/addhuZUÖB

 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2@8Ä@ÄHÄb>gradient_tape/model_1/batch_normalization_18/moments/truediv_1huZUÖB
G
!Equal_GPU_DT_INT64_DT_BOOL_kernel*Ä28Ä@ÄHÄbEqualhuZUÖB
o
'GreaterEqual_GPU_DT_HALF_DT_BOOL_kernel*Ä28Ä@ÄHÄb'model_1/dropout_10/dropout/GreaterEqualhuZUÖB
o
'GreaterEqual_GPU_DT_HALF_DT_BOOL_kernel*Ä28Ä@ÄHÄb'model_1/dropout_11/dropout/GreaterEqualhuZUÖB
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_11hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_13hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_14hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_17hu  »B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄb
IsFinite_2hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_22hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_25hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_29hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_30hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_35hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_37hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_41hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_45hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_49hu  »B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄb
IsFinite_5hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_51hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_53hu  »B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄb
IsFinite_6hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_61hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_54hu  »B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄb
IsFinite_4hu  »B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄb
IsFinite_8hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2@8Ä@ÄHÄbIsFinite_16hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2@8Ä@ÄHÄbIsFinite_24hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2P8Ä@ÄHÄbIsFinite_32hu  »B
D
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbMulhuZUÖB
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb conv2d_27/kernel/Regularizer/mulhuZUÖB
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb conv2d_33/kernel/Regularizer/mulhuZUÖB
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb conv2d_37/kernel/Regularizer/mulhuZUÖB
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb conv2d_40/kernel/Regularizer/mulhuZUÖB
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb conv2d_45/kernel/Regularizer/mulhuZUÖB
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb conv2d_46/kernel/Regularizer/mulhuZUÖB
_
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbdense_3/kernel/Regularizer/mulhuZUÖB
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb.gradient_tape/conv2d_27/kernel/Regularizer/MulhuZUÖB
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb4gradient_tape/conv2d_27/kernel/Regularizer/mul/Mul_1huZUÖB
Å
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb@gradient_tape/model_1/batch_normalization_19/batchnorm/mul/Mul_1huZUÖB
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb2model_1/batch_normalization_19/AssignMovingAvg/mulhuZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_11huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_13huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_15huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_16huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_21huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_23huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_24huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_25huZUÖB
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_3huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_32huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_39huZUÖB
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_4huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_40huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_41huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_47huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_48huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_53huZUÖB
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
q
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb0gradient_tape/conv2d_27/kernel/Regularizer/Mul_1huZUÖB

 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb>gradient_tape/model_1/batch_normalization_18/batchnorm/mul/MulhuZUÖB
É
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbBgradient_tape/model_1/batch_normalization_18/batchnorm/mul_2/Mul_1huZUÖB
Å
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb@gradient_tape/model_1/batch_normalization_19/batchnorm/mul_2/MulhuZUÖB
É
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbBgradient_tape/model_1/batch_normalization_19/batchnorm/mul_2/Mul_1huZUÖB
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb.model_1/batch_normalization_18/batchnorm/mul_2huZUÖB
m
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb,model_1/batch_normalization_19/batchnorm/mulhuZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_60huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_64huZUÖB
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb.gradient_tape/conv2d_29/kernel/Regularizer/MulhuZUÖB
q
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb0gradient_tape/conv2d_29/kernel/Regularizer/Mul_1huZUÖB
m
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb,gradient_tape/dense_5/kernel/Regularizer/MulhuZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_66huZUÖB
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb.gradient_tape/dense_5/kernel/Regularizer/Mul_1huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_10huZUÖB
q
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb0gradient_tape/conv2d_30/kernel/Regularizer/Mul_1huZUÖB
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb.gradient_tape/conv2d_33/kernel/Regularizer/MulhuZUÖB
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb.gradient_tape/conv2d_36/kernel/Regularizer/MulhuZUÖB
y
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb8gradient_tape/model_1/batch_normalization_18/moments/MulhuZUÖB
y
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb8gradient_tape/model_1/batch_normalization_19/moments/MulhuZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_18huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_26huZUÖB
q
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb0gradient_tape/conv2d_33/kernel/Regularizer/Mul_1huZUÖB
q
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb0gradient_tape/conv2d_36/kernel/Regularizer/Mul_1huZUÖB
É
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbBgradient_tape/model_1/batch_normalization_18/batchnorm/mul_1/Mul_1huZUÖB
É
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbBgradient_tape/model_1/batch_normalization_19/batchnorm/mul_1/Mul_1huZUÖB
{
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb:gradient_tape/model_1/batch_normalization_19/moments/mul_1huZUÖB
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb.gradient_tape/conv2d_39/kernel/Regularizer/MulhuZUÖB
q
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb0gradient_tape/conv2d_39/kernel/Regularizer/Mul_1huZUÖB
Å
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2@8Ä@ÄHÄb@gradient_tape/model_1/batch_normalization_18/batchnorm/mul_1/MulhuZUÖB
Å
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2@8Ä@ÄHÄb@gradient_tape/model_1/batch_normalization_19/batchnorm/mul_1/MulhuZUÖB
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2@8Ä@ÄHÄb.model_1/batch_normalization_18/batchnorm/mul_1huZUÖB
k
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä28Ä@ÄHÄb,gradient_tape/model_1/dropout_10/dropout/MulhuZUÖB
m
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä28Ä@ÄHÄb.gradient_tape/model_1/dropout_10/dropout/Mul_1huZUÖB
m
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä28Ä@ÄHÄb.gradient_tape/model_1/dropout_11/dropout/Mul_1huZUÖB
]
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä28Ä@ÄHÄbmodel_1/dropout_10/dropout/MulhuZUÖB
_
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä28Ä@ÄHÄb model_1/dropout_11/dropout/Mul_1huZUÖB

 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb>gradient_tape/model_1/batch_normalization_18/batchnorm/sub/Neghu  »B
q
"Rsqrt_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb.model_1/batch_normalization_18/batchnorm/Rsqrthu  »B
q
"Rsqrt_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb.model_1/batch_normalization_19/batchnorm/Rsqrthu  »B
g
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb#conv2d_30/kernel/Regularizer/SquarehuZUÖB
g
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb#conv2d_33/kernel/Regularizer/SquarehuZUÖB
g
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb#conv2d_36/kernel/Regularizer/SquarehuZUÖB
g
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb#conv2d_39/kernel/Regularizer/SquarehuZUÖB
s
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb2model_1/batch_normalization_18/AssignMovingAvg/subhuZUÖB
u
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb4model_1/batch_normalization_18/AssignMovingAvg_1/subhuZUÖB
y
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2@8Ä@ÄHÄb8gradient_tape/model_1/batch_normalization_18/moments/subhuZUÖB
y
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2@8Ä@ÄHÄb8gradient_tape/model_1/batch_normalization_19/moments/subhuZUÖB
µ
htensorflow::functor::ReluGradHalfKernelVector(Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä28Ä@ÄHÄb,gradient_tape/model_1/activation_19/ReluGradhu  »B
Æ
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_1/conv2d_28/Conv2D/CasthuZUÖB
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_1/conv2d_29/BiasAdd/CasthuZUÖB
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_1/conv2d_31/BiasAdd/CasthuZUÖB
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_1/conv2d_32/BiasAdd/CasthuZUÖB
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_1/conv2d_41/BiasAdd/CasthuZUÖB
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_1/conv2d_44/BiasAdd/CasthuZUÖB
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_1/conv2d_46/BiasAdd/CasthuZUÖB
≠
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_1/dense_5/BiasAdd/CasthuZUÖB
Æ
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_1/conv2d_29/Conv2D/CasthuZUÖB
¨
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_1/dense_5/MatMul/CasthuZUÖB
«
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄb6gradient_tape/model_1/batch_normalization_19/Cast/CasthuZUÖB
∂
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄb%model_1/batch_normalization_18/Cast_1huZUÖB
Æ
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä@ÄHÄbmodel_1/conv2d_31/Conv2D/CasthuZUÖB
Æ
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä@ÄHÄbmodel_1/conv2d_32/Conv2D/CasthuZUÖB
Æ
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä@ÄHÄbmodel_1/conv2d_42/Conv2D/CasthuZUÖB
£	
Ávoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄbmodel_1/activation_19/ReluhuZUÖB
£	
Ávoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄbmodel_1/activation_20/ReluhuZUÖB
‡
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä28Ä@ÄHÄb;gradient_tape/categorical_crossentropy/weighted_loss/Tile_1huZUÖB
ì
≈void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb,categorical_crossentropy/weighted_loss/valuehuZUÖB
Ω
€void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_rsqrt_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_rsqrt_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb@gradient_tape/model_1/batch_normalization_18/batchnorm/RsqrtGradhuZUÖB
Ω
€void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_rsqrt_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_rsqrt_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb@gradient_tape/model_1/batch_normalization_19/batchnorm/RsqrtGradhuZUÖB
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb1gradient_tape/model_1/conv2d_28/BiasAdd/Cast/CasthuZUÖB
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb1gradient_tape/model_1/conv2d_31/BiasAdd/Cast/CasthuZUÖB
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb1gradient_tape/model_1/conv2d_32/BiasAdd/Cast/CasthuZUÖB
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb1gradient_tape/model_1/conv2d_33/BiasAdd/Cast/CasthuZUÖB
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb1gradient_tape/model_1/conv2d_38/BiasAdd/Cast/CasthuZUÖB
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb1gradient_tape/model_1/conv2d_41/BiasAdd/Cast/CasthuZUÖB
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb1gradient_tape/model_1/conv2d_44/BiasAdd/Cast/CasthuZUÖB
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb1gradient_tape/model_1/conv2d_47/BiasAdd/Cast/CasthuZUÖB
¥
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb/gradient_tape/model_1/dense_4/BiasAdd/Cast/CasthuZUÖB
≥
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb.gradient_tape/model_1/dense_5/MatMul/Cast/CasthuZUÖB
µ
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb0gradient_tape/model_1/conv2d_30/Conv2D/Cast/CasthuZUÖB
Ω
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄb8gradient_tape/model_1/batch_normalization_19/Cast_1/CasthuZUÖB
µ
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄb0gradient_tape/model_1/conv2d_33/Conv2D/Cast/CasthuZUÖB
µ
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2P8Ä@ÄHÄb0gradient_tape/model_1/conv2d_39/Conv2D/Cast/CasthuZUÖB
˝
’void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbCast_7huZUÖB
˚
”void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbCast_8huZUÖB
∞
”void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb;gradient_tape/model_1/batch_normalization_18/moments/Cast_1huZUÖB
∞
”void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb;gradient_tape/model_1/batch_normalization_19/moments/Cast_1huZUÖB
¡
Òvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb.model_1/batch_normalization_18/AssignMovingAvghuZUÖB
√
Òvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb0model_1/batch_normalization_18/AssignMovingAvg_1huZUÖB
√
Òvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb0model_1/batch_normalization_19/AssignMovingAvg_1huZUÖB
ß
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbAdam/gradients/AddN_10huZUÖB
§
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbAdam/gradients/AddNhuZUÖB
ß
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbAdam/gradients/AddN_12huZUÖB
ß
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbAdam/gradients/AddN_13huZUÖB
ß
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄbAdam/gradients/AddN_16huZUÖB
ß
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2P8Ä@ÄHÄbAdam/gradients/AddN_22huZUÖB
ˆ	
ø	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄbAdam/gradients/AddN_1huZUÖB
ˆ	
ø	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄbAdam/gradients/AddN_3huZUÖB
È
üvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long) *Ä28Ä@ÄHÄb(ArithmeticOptimizer/AddOpsRewrite_AddN_1huZUÖB
“
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä28Ä@ÄHÄb-gradient_tape/dense_5/kernel/Regularizer/TilehuZUÖB
Â
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2@8Ä@ÄHÄb@gradient_tape/model_1/batch_normalization_18/moments/BroadcastTohuZUÖB
Á
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2@8Ä@ÄHÄbBgradient_tape/model_1/batch_normalization_18/moments/BroadcastTo_1huZUÖB
Â
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2@8Ä@ÄHÄb@gradient_tape/model_1/batch_normalization_19/moments/BroadcastTohuZUÖB
Á
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2@8Ä@ÄHÄbBgradient_tape/model_1/batch_normalization_19/moments/BroadcastTo_1huZUÖB
¡	
Ávoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_squared_difference_op<float>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_squared_difference_op<float>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, int) *Ä2@8Ä@ÄHÄb8model_1/batch_normalization_18/moments/SquaredDifferencehuZUÖB
ˇ	
£	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)&*Ä28Ä@ÄHÄb:categorical_crossentropy/softmax_cross_entropy_with_logitshuZUÖB
‘
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä28Ä@ÄHÄb/gradient_tape/conv2d_28/kernel/Regularizer/TilehuZUÖB
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
à
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2$8Ä@ÄHÄbAll_10hu  »B
à
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2(8Ä@ÄHÄbAll_40hu  »B
à
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä228Ä@ÄHÄbAll_12hu  »B
Ö
€void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *Ä28Ä@ÄHÄbAll_12hu¶™¶B
Ö
€void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *Ä28Ä@ÄHÄbAll_20hu¶™¶B
Ö
€void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *Ä28Ä@ÄHÄbAll_26hu¶™¶B
Ö
€void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *Ä28Ä@ÄHÄbAll_28hu¶™¶B
Ö
€void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *Ä28Ä@ÄHÄbAll_34hu¶™¶B
Ö
€void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *Ä28Ä@ÄHÄbAll_36hu¶™¶B
Ö
€void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *Ä28Ä@ÄHÄbAll_40hu¶™¶B
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
€void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *Ä28Ä@ÄHÄbAll_60hu¶™¶B
Ω
˘void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float>, float>(float*, float*, int, tensorflow::functor::Sum<float>, float)0*Ä28Ä@ÄHÄb conv2d_32/kernel/Regularizer/Sumhu  »B
Ω
˘void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float>, float>(float*, float*, int, tensorflow::functor::Sum<float>, float)0*Ä28Ä@ÄHÄb conv2d_38/kernel/Regularizer/Sumhu  »B
Ω
˘void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float>, float>(float*, float*, int, tensorflow::functor::Sum<float>, float)0*Ä28Ä@ÄHÄb conv2d_40/kernel/Regularizer/Sumhu  »B
Ω
˘void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float>, float>(float*, float*, int, tensorflow::functor::Sum<float>, float)0*Ä28Ä@ÄHÄb conv2d_41/kernel/Regularizer/Sumhu  »B
Ω
˘void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float>, float>(float*, float*, int, tensorflow::functor::Sum<float>, float)0*Ä28Ä@ÄHÄb conv2d_43/kernel/Regularizer/Sumhu  »B
Ω
˘void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float>, float>(float*, float*, int, tensorflow::functor::Sum<float>, float)0*Ä28Ä@ÄHÄb conv2d_44/kernel/Regularizer/Sumhu  »B
Ω
˘void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float>, float>(float*, float*, int, tensorflow::functor::Sum<float>, float)0*Ä28Ä@ÄHÄb conv2d_45/kernel/Regularizer/Sumhu  »B
Ω
˘void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float>, float>(float*, float*, int, tensorflow::functor::Sum<float>, float)0*Ä28Ä@ÄHÄb conv2d_46/kernel/Regularizer/Sumhu  »B
Ω
˘void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float>, float>(float*, float*, int, tensorflow::functor::Sum<float>, float)0*Ä28Ä@ÄHÄb conv2d_47/kernel/Regularizer/Sumhu  »B
ª
˘void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float>, float>(float*, float*, int, tensorflow::functor::Sum<float>, float)0*Ä28Ä@ÄHÄbdense_4/kernel/Regularizer/Sumhu  »B
ﬁ
§void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*) *Ä28Ä@ÄHÄXbmodel_1/dense_5/MatMulhu  »B
Ñ
§void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*) *Ä2Ä8Ä@ÄHÄXb;gradient_tape/model_1/conv2d_36/Conv2D/Conv2DBackpropFilterhu  »B
Ç
¢void splitKreduce_kernel<float, __half, float, __half>(cublasSplitKParams<float>, float const*, __half const*, __half*, float const*, float const*, __half const*) *Ä2Ä8Ä@ÄHÄXb;gradient_tape/model_1/conv2d_30/Conv2D/Conv2DBackpropFilterhu  »B
ª
évoid tensorflow::(anonymous namespace)::concat_fixed_kernel<bool, int>(tensorflow::GpuDeviceArrayStruct<bool const*, 8>, int, int, int, bool*)*B28Ä@ÄHÄbAll_66/inputhu∞öB
§
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä28Ä@ÄHÄbmodel_1/dense_5/BiasAddhuZUÖB
§
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2@8Ä@ÄHÄbmodel_1/dense_3/BiasAddhuZUÖB
§
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2@8Ä@ÄHÄbmodel_1/dense_4/BiasAddhuZUÖB
í
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb;cond_1/then/_10/cond_1/Adam/Adam/update_1/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb<cond_1/then/_10/cond_1/Adam/Adam/update_11/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb<cond_1/then/_10/cond_1/Adam/Adam/update_13/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb<cond_1/then/_10/cond_1/Adam/Adam/update_19/ResourceApplyAdamhuZUÖB
í
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb;cond_1/then/_10/cond_1/Adam/Adam/update_2/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb<cond_1/then/_10/cond_1/Adam/Adam/update_22/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb<cond_1/then/_10/cond_1/Adam/Adam/update_25/ResourceApplyAdamhuZUÖB
í
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb;cond_1/then/_10/cond_1/Adam/Adam/update_3/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb<cond_1/then/_10/cond_1/Adam/Adam/update_31/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb<cond_1/then/_10/cond_1/Adam/Adam/update_33/ResourceApplyAdamhuZUÖB
í
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb;cond_1/then/_10/cond_1/Adam/Adam/update_6/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb<cond_1/then/_10/cond_1/Adam/Adam/update_65/ResourceApplyAdamhuZUÖB
í
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb;cond_1/then/_10/cond_1/Adam/Adam/update_7/ResourceApplyAdamhuZUÖB
í
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb;cond_1/then/_10/cond_1/Adam/Adam/update_9/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb<cond_1/then/_10/cond_1/Adam/Adam/update_55/ResourceApplyAdamhuZUÖB
’
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAllhu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_11hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_14hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_15hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_21hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_22hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_27hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_30hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_31hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_35hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_37hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_38hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_39hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_41hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_43hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_45hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_47hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_58hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_62hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_63hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_65hu  »B
◊
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_9hu  »B
Ü
¬void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*Ä28Ä@ÄHÄb conv2d_27/kernel/Regularizer/Sumhu  »B
Ü
¬void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*Ä28Ä@ÄHÄb conv2d_28/kernel/Regularizer/Sumhu  »B
Ü
¬void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*Ä2 8Ä@ÄHÄb conv2d_30/kernel/Regularizer/Sumhu  »B
Ü
¬void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*Ä2 8Ä@ÄHÄb conv2d_36/kernel/Regularizer/Sumhu  »B
Ü
¬void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*Ä2 8Ä@ÄHÄb conv2d_39/kernel/Regularizer/Sumhu  »B
Ñ
¬void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*Ä2 8Ä@ÄHÄbdense_5/kernel/Regularizer/Sumhu  »B
à
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2 8Ä@ÄHÄb3gradient_tape/model_1/conv2d_27/BiasAdd/BiasAddGradhuZUÖB
à
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2@8Ä@ÄHÄb3gradient_tape/model_1/conv2d_29/BiasAdd/BiasAddGradhuZUÖB
â
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Ä@ÄHÄb3gradient_tape/model_1/conv2d_32/BiasAdd/BiasAddGradhuZUÖB
â
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Ä@ÄHÄb3gradient_tape/model_1/conv2d_35/BiasAdd/BiasAddGradhuZUÖB
ÿ
±void tensorflow::functor::CleanupSegments<bool*, bool*, tensorflow::functor::And>(bool*, bool*, int, int, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type)* 28Ä@ÄHÄbAll_16huMUB
ÿ
±void tensorflow::functor::CleanupSegments<bool*, bool*, tensorflow::functor::And>(bool*, bool*, int, int, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type)* 28Ä@ÄHÄbAll_24huMUB
ÿ
±void tensorflow::functor::CleanupSegments<bool*, bool*, tensorflow::functor::And>(bool*, bool*, int, int, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type)* 28Ä@ÄHÄbAll_32huMUB
Ü
≈void tensorflow::functor::CleanupSegments<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)* 28Ä@ÄHÄb conv2d_29/kernel/Regularizer/SumhuMUB
Ü
≈void tensorflow::functor::CleanupSegments<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)* 28Ä@ÄHÄb conv2d_30/kernel/Regularizer/SumhuMUB
Ü
≈void tensorflow::functor::CleanupSegments<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)* 28Ä@ÄHÄb conv2d_33/kernel/Regularizer/SumhuMUB
Ü
≈void tensorflow::functor::CleanupSegments<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)* 28Ä@ÄHÄb conv2d_39/kernel/Regularizer/SumhuMUB
ô
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2@8Ä@ÄHÄb1gradient_tape/model_1/dense_3/BiasAdd/BiasAddGradhuZUÖB
©
√void tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)Ä!*  2@8Ä@ÄHÄbBgradient_tape/model_1/batch_normalization_18/batchnorm/add_1/Sum_1huZUÖB
©
√void tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)Ä!*  2@8Ä@ÄHÄbBgradient_tape/model_1/batch_normalization_18/batchnorm/mul_1/Sum_1huZUÖB
–
Åvoid tensorflow::functor::ColumnReduceKernel<float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, tensorflow::functor::Sum<float> >(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)Ä!*  2@8Ä@ÄHÄb+model_1/batch_normalization_18/moments/meanhuZUÖB
‘
Åvoid tensorflow::functor::ColumnReduceKernel<float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, tensorflow::functor::Sum<float> >(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)Ä!*  2@8Ä@ÄHÄb/model_1/batch_normalization_18/moments/variancehuZUÖB
”
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä28Ä@ÄHÄXbmodel_1/conv2d_28/Conv2DhuZUÖB
”
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä28Ä@ÄHÄXbmodel_1/conv2d_29/Conv2DhuZUÖB
ˆ
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä28Ä@ÄHÄXb;gradient_tape/model_1/conv2d_28/Conv2D/Conv2DBackpropFilterhuZUÖB
ˆ
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä28Ä@ÄHÄXb;gradient_tape/model_1/conv2d_29/Conv2D/Conv2DBackpropFilterhuZUÖB
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Å@ÅHÅbIsFinite_57hu  »B
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Å@ÅHÅb conv2d_34/kernel/Regularizer/mulhuZUÖB
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Å@ÅHÅb conv2d_42/kernel/Regularizer/mulhuZUÖB
Å
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Å@ÅHÅb@gradient_tape/model_1/batch_normalization_18/batchnorm/mul/Mul_1huZUÖB

 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Å@ÅHÅb>gradient_tape/model_1/batch_normalization_19/batchnorm/mul/MulhuZUÖB
ñ
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Å@ÅHÅbêgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/Cast_2/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_CasthuZUÖB
µ
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Å@ÅHÅb0gradient_tape/model_1/conv2d_28/Conv2D/Cast/CasthuZUÖB
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Å@ÅHÅb1gradient_tape/model_1/conv2d_35/BiasAdd/Cast/CasthuZUÖB
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Å@ÅHÅb1gradient_tape/model_1/conv2d_46/BiasAdd/Cast/CasthuZUÖB
b
"AddV2_GPU_DT_INT64_DT_INT64_kernel*Ä28Ä@ÄHÄbcond_1/then/_10/cond_1/Adam/addhuZUÖB
}
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2@8Ä@ÄHÄb<gradient_tape/model_1/batch_normalization_18/moments/truedivhuZUÖB
}
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2@8Ä@ÄHÄb<gradient_tape/model_1/batch_normalization_19/moments/truedivhuZUÖB

 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2@8Ä@ÄHÄb>gradient_tape/model_1/batch_normalization_19/moments/truediv_1huZUÖB
g
(GreaterEqual_GPU_DT_INT64_DT_BOOL_kernel*Ä28Ä@ÄHÄbcond/then/_0/cond/GreaterEqualhuZUÖB
M
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinitehu  »B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄb
IsFinite_1hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_15hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_19hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_21hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_23hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_27hu  »B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄb
IsFinite_3hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_31hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_33hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_38hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_39hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_43hu  »B
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
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_62hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_63hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_55hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_64hu  »B
P
%LogicalAnd_GPU_DT_BOOL_DT_BOOL_kernel*Ä28Ä@ÄHÄb
LogicalAndhuZUÖB
ç
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbLgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mulhuZUÖB
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb conv2d_28/kernel/Regularizer/mulhuZUÖB
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb conv2d_29/kernel/Regularizer/mulhuZUÖB
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb conv2d_31/kernel/Regularizer/mulhuZUÖB
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb conv2d_32/kernel/Regularizer/mulhuZUÖB
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb conv2d_35/kernel/Regularizer/mulhuZUÖB
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb conv2d_36/kernel/Regularizer/mulhuZUÖB
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb conv2d_38/kernel/Regularizer/mulhuZUÖB
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb conv2d_39/kernel/Regularizer/mulhuZUÖB
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb conv2d_41/kernel/Regularizer/mulhuZUÖB
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb conv2d_43/kernel/Regularizer/mulhuZUÖB
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb conv2d_44/kernel/Regularizer/mulhuZUÖB
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb conv2d_47/kernel/Regularizer/mulhuZUÖB
_
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbdense_4/kernel/Regularizer/mulhuZUÖB
_
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbdense_5/kernel/Regularizer/mulhuZUÖB
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb.gradient_tape/conv2d_28/kernel/Regularizer/MulhuZUÖB
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb2model_1/batch_normalization_18/AssignMovingAvg/mulhuZUÖB
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb4model_1/batch_normalization_18/AssignMovingAvg_1/mulhuZUÖB
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb4model_1/batch_normalization_19/AssignMovingAvg_1/mulhuZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_17huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_19huZUÖB
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_2huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_27huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_29huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_31huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_33huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_35huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_37huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_43huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_45huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_49huZUÖB
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_5huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_51huZUÖB
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
q
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb0gradient_tape/conv2d_28/kernel/Regularizer/Mul_1huZUÖB
Å
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb@gradient_tape/model_1/batch_normalization_18/batchnorm/mul_2/MulhuZUÖB
m
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb,model_1/batch_normalization_18/batchnorm/mulhuZUÖB
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb.model_1/batch_normalization_19/batchnorm/mul_2huZUÖB
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_6huZUÖB
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb.gradient_tape/conv2d_30/kernel/Regularizer/MulhuZUÖB
{
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb:gradient_tape/model_1/batch_normalization_18/moments/mul_1huZUÖB
k
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä28Ä@ÄHÄb,gradient_tape/model_1/dropout_11/dropout/MulhuZUÖB
_
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä28Ä@ÄHÄb model_1/dropout_10/dropout/Mul_1huZUÖB
]
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä28Ä@ÄHÄbmodel_1/dropout_11/dropout/MulhuZUÖB

 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb>gradient_tape/model_1/batch_normalization_19/batchnorm/sub/Neghu  »B
`
 Pow_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbcond_1/then/_10/cond_1/Adam/PowhuZUÖB
b
 Pow_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb!cond_1/then/_10/cond_1/Adam/Pow_1huZUÖB
g
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb#conv2d_27/kernel/Regularizer/SquarehuZUÖB
g
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb#conv2d_28/kernel/Regularizer/SquarehuZUÖB
g
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb#conv2d_29/kernel/Regularizer/SquarehuZUÖB
e
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb!dense_5/kernel/Regularizer/SquarehuZUÖB
m
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb,model_1/batch_normalization_18/batchnorm/subhuZUÖB
s
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb2model_1/batch_normalization_19/AssignMovingAvg/subhuZUÖB
u
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb4model_1/batch_normalization_19/AssignMovingAvg_1/subhuZUÖB
m
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb,model_1/batch_normalization_19/batchnorm/subhuZUÖB
µ
htensorflow::functor::ReluGradHalfKernelVector(Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä28Ä@ÄHÄb,gradient_tape/model_1/activation_20/ReluGradhu  »B
Æ
Ìvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄbmodel_1/dropout_10/dropout/CasthuZUÖB
Æ
Ìvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄbmodel_1/dropout_11/dropout/CasthuZUÖB
ï
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbCasthuZUÖB
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
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_1/conv2d_27/BiasAdd/CasthuZUÖB
Æ
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_1/conv2d_27/Conv2D/CasthuZUÖB
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_1/conv2d_28/BiasAdd/CasthuZUÖB
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_1/conv2d_30/BiasAdd/CasthuZUÖB
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_1/conv2d_33/BiasAdd/CasthuZUÖB
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_1/conv2d_34/BiasAdd/CasthuZUÖB
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_1/conv2d_35/BiasAdd/CasthuZUÖB
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_1/conv2d_36/BiasAdd/CasthuZUÖB
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_1/conv2d_37/BiasAdd/CasthuZUÖB
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_1/conv2d_38/BiasAdd/CasthuZUÖB
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_1/conv2d_39/BiasAdd/CasthuZUÖB
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_1/conv2d_40/BiasAdd/CasthuZUÖB
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_1/conv2d_42/BiasAdd/CasthuZUÖB
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_1/conv2d_43/BiasAdd/CasthuZUÖB
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_1/conv2d_45/BiasAdd/CasthuZUÖB
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_1/conv2d_47/BiasAdd/CasthuZUÖB
≠
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_1/dense_3/BiasAdd/CasthuZUÖB
≠
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_1/dense_4/BiasAdd/CasthuZUÖB
Æ
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_1/conv2d_30/Conv2D/CasthuZUÖB
«
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄb6gradient_tape/model_1/batch_normalization_18/Cast/CasthuZUÖB
∂
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄb%model_1/batch_normalization_19/Cast_1huZUÖB
Æ
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄbmodel_1/conv2d_33/Conv2D/CasthuZUÖB
Æ
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄbmodel_1/conv2d_36/Conv2D/CasthuZUÖB
Æ
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2P8Ä@ÄHÄbmodel_1/conv2d_39/Conv2D/CasthuZUÖB
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
Ç
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb}categorical_crossentropy/softmax_cross_entropy_with_logits/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Cast_1huZUÖB
ƒ
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb?categorical_crossentropy/softmax_cross_entropy_with_logits/CasthuZUÖB
∆
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbAcategorical_crossentropy/softmax_cross_entropy_with_logits/Cast_1huZUÖB
Ï
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbgcategorical_crossentropy/weighted_loss/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_CasthuZUÖB
≈
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb@gradient_tape/categorical_crossentropy/weighted_loss/Cast_1/CasthuZUÖB
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb1gradient_tape/model_1/conv2d_27/BiasAdd/Cast/CasthuZUÖB
µ
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb0gradient_tape/model_1/conv2d_27/Conv2D/Cast/CasthuZUÖB
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb1gradient_tape/model_1/conv2d_29/BiasAdd/Cast/CasthuZUÖB
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb1gradient_tape/model_1/conv2d_30/BiasAdd/Cast/CasthuZUÖB
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb1gradient_tape/model_1/conv2d_34/BiasAdd/Cast/CasthuZUÖB
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb1gradient_tape/model_1/conv2d_36/BiasAdd/Cast/CasthuZUÖB
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb1gradient_tape/model_1/conv2d_37/BiasAdd/Cast/CasthuZUÖB
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb1gradient_tape/model_1/conv2d_39/BiasAdd/Cast/CasthuZUÖB
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb1gradient_tape/model_1/conv2d_40/BiasAdd/Cast/CasthuZUÖB
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb1gradient_tape/model_1/conv2d_42/BiasAdd/Cast/CasthuZUÖB
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb1gradient_tape/model_1/conv2d_43/BiasAdd/Cast/CasthuZUÖB
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb1gradient_tape/model_1/conv2d_45/BiasAdd/Cast/CasthuZUÖB
¥
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb/gradient_tape/model_1/dense_5/BiasAdd/Cast/CasthuZUÖB
¥
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb/gradient_tape/model_1/dense_3/BiasAdd/Cast/CasthuZUÖB
µ
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb0gradient_tape/model_1/conv2d_29/Conv2D/Cast/CasthuZUÖB
µ
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄb0gradient_tape/model_1/conv2d_36/Conv2D/Cast/CasthuZUÖB
®
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄb#model_1/batch_normalization_18/CasthuZUÖB
®
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄb#model_1/batch_normalization_19/CasthuZUÖB
˚
”void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbCast_2huZUÖB
≠
”void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb8categorical_crossentropy/weighted_loss/num_elements/CasthuZUÖB
Æ
”void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb9gradient_tape/model_1/batch_normalization_18/moments/CasthuZUÖB
Æ
”void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb9gradient_tape/model_1/batch_normalization_19/moments/CasthuZUÖB
¡
Òvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb.model_1/batch_normalization_19/AssignMovingAvghuZUÖB
ß
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbAdam/gradients/AddN_11huZUÖB
¶
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbAdam/gradients/AddN_2huZUÖB
ß
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbAdam/gradients/AddN_33huZUÖB
ò
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbAssignAddVariableOphuZUÖB
ö
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbAssignAddVariableOp_1huZUÖB
ö
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbAssignAddVariableOp_2huZUÖB