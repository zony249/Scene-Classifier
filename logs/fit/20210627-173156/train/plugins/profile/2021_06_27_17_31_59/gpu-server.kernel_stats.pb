
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�2R8ŗ�@ŗ�Hŗ�b<cond_1/then/_10/cond_1/Adam/Adam/update_56/ResourceApplyAdamhuZU�B
�
�_ZN13cutlass_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi64ELi256ELi32EEENS4_51Conv2dWgradOutputGradientTileAccessIteratorAnalyticINS_11MatrixShapeILi64ELi32EEENS_6half_tENS_9transform29PitchLinearWarpRakedThreadMapINS_6layout16PitchLinearShapeILi64ELi32EEELi128ENSG_ILi8ELi4EEELi8EEEEENSD_11threadblock25RegularTileAccessIteratorISB_SC_NSF_40ColumnMajorTensorOpMultiplicandCongruousILi16ELi64EEELi1ESJ_Li16EEELNS_4arch14CacheOperation4KindE0ENS4_47Conv2dWgradActivationTileAccessIteratorAnalyticINSA_ILi32ELi256EEESC_NSE_INSG_ILi256ELi32EEELi128ESI_Li8EEEEENSM_ISU_SC_NSF_37RowMajorTensorOpMultiplicandCongruousILi16ELi64EEELi0ESW_Li16EEELSS_0ENS6_11threadblock9MmaPolicyINS6_4warp11MmaTensorOpINS7_ILi64ELi64ELi32EEESC_SO_SC_SZ_fNSF_8RowMajorENS13_17MmaTensorOpPolicyINSQ_3MmaINS7_ILi16ELi8ELi16EEELi32ESC_S16_SC_NSF_11ColumnMajorEfS16_NSQ_13OpMultiplyAddEEENSA_ILi1ELi1EEEEELi1ELb0EbEENSA_ILi0ELi0EEES1G_Li1EEELi4EbEENS_8epilogue11threadblock8EpilogueIS8_S1F_Li1ENS1K_22PredicatedTileIteratorINS1K_26OutputTileOptimalThreadMapINS1K_15OutputTileShapeILi256ELi8ELi1ELi1ELi1EEENS1O_ILi1ELi8ELi1ELi1ELi8EEELi128ELi4ELi32EEEfEENS1J_4warp24FragmentIteratorTensorOpIS15_S19_fNS_5ArrayIfLi4ELb1EEES16_EENS1T_20TileIteratorTensorOpIS15_S19_fS16_EENS1K_18SharedLoadIteratorINS1R_18CompactedThreadMapEfLi16EEENS1J_6thread17LinearCombinationIfLi4EffLNS_15FloatRoundStyleE2EEENSA_ILi0ELi8EEEEENS11_30GemmIdentityThreadblockSwizzleILi4EEELNS1_8OperatorE2ENS1_17Conv2dProblemSizeEEEEEvNT_6ParamsE� ��*�22)8ݧ�@ݧ�Hݧ�Xb;gradient_tape/model_3/conv2d_83/Conv2D/Conv2DBackpropFilterh
�
|sm80_xmma_fprop_implicit_gemm_f16f16_f16f32_f32_nhwckrsc_nhwc_tilesize256x64x32_stage3_warpsize4x1x1_g1_tensor16x8x16_kernel� ��*�2�8���@���H���PXbmodel_3/conv2d_83/Conv2Dh
�
�void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride>(cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride::Params)� ��*�2�8��@��H��Xb:gradient_tape/model_3/conv2d_83/Conv2D/Conv2DBackpropInputh
�
�void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride>(cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride::Params)� ��*�2�8���@���H���Xb:gradient_tape/model_3/conv2d_89/Conv2D/Conv2DBackpropInputh
�
�void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3>(cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3::Params)� ��*�2�8鏋@鏋H鏋Xbmodel_3/conv2d_89/Conv2Dh
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8�ϊ@�ϊH�ϊbAdam/gradients/AddN_31huZU�B
w
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2��8韄@韄H韄b.gradient_tape/dense_9/kernel/Regularizer/Mul_1huZU�B
�
�void xmma_cudnn::gemm::kernel<xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, xmma_cudnn::Col, 256, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 3> >(xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, xmma_cudnn::Col, 256, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 3>::Params)� ��*�2$	8�ǁ@�ǁH�ǁPXb;gradient_tape/model_3/conv2d_82/Conv2D/Conv2DBackpropFilterh
�
�void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3>(cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3::Params)� ��*�2�8��@��H��Xbmodel_3/conv2d_82/Conv2Dh
�
�_ZN13cutlass_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi64ELi256ELi32EEENS4_51Conv2dWgradOutputGradientTileAccessIteratorAnalyticINS_11MatrixShapeILi64ELi32EEENS_6half_tENS_9transform29PitchLinearWarpRakedThreadMapINS_6layout16PitchLinearShapeILi64ELi32EEELi128ENSG_ILi8ELi4EEELi8EEEEENSD_11threadblock25RegularTileAccessIteratorISB_SC_NSF_40ColumnMajorTensorOpMultiplicandCongruousILi16ELi64EEELi1ESJ_Li16EEELNS_4arch14CacheOperation4KindE0ENS4_47Conv2dWgradActivationTileAccessIteratorAnalyticINSA_ILi32ELi256EEESC_NSE_INSG_ILi256ELi32EEELi128ESI_Li8EEEEENSM_ISU_SC_NSF_37RowMajorTensorOpMultiplicandCongruousILi16ELi64EEELi0ESW_Li16EEELSS_0ENS6_11threadblock9MmaPolicyINS6_4warp11MmaTensorOpINS7_ILi64ELi64ELi32EEESC_SO_SC_SZ_fNSF_8RowMajorENS13_17MmaTensorOpPolicyINSQ_3MmaINS7_ILi16ELi8ELi16EEELi32ESC_S16_SC_NSF_11ColumnMajorEfS16_NSQ_13OpMultiplyAddEEENSA_ILi1ELi1EEEEELi1ELb0EbEENSA_ILi0ELi0EEES1G_Li1EEELi4EbEENS_8epilogue11threadblock8EpilogueIS8_S1F_Li1ENS1K_22PredicatedTileIteratorINS1K_26OutputTileOptimalThreadMapINS1K_15OutputTileShapeILi256ELi8ELi1ELi1ELi1EEENS1O_ILi1ELi8ELi1ELi1ELi8EEELi128ELi4ELi32EEEfEENS1J_4warp24FragmentIteratorTensorOpIS15_S19_fNS_5ArrayIfLi4ELb1EEES16_EENS1T_20TileIteratorTensorOpIS15_S19_fS16_EENS1K_18SharedLoadIteratorINS1R_18CompactedThreadMapEfLi16EEENS1J_6thread17LinearCombinationIfLi4EffLNS_15FloatRoundStyleE2EEENSA_ILi0ELi8EEEEENS11_30GemmIdentityThreadblockSwizzleILi4EEELNS1_8OperatorE2ENS1_17Conv2dProblemSizeEEEEEvNT_6ParamsE� ��*�2
8��@��H��Xb;gradient_tape/model_3/conv2d_89/Conv2D/Conv2DBackpropFilterh
�
�void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride>(cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride::Params)� ��*�2�8���@���H���Xb:gradient_tape/model_3/conv2d_82/Conv2D/Conv2DBackpropInputh
�
�void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3>(cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3::Params)� ��*�2�8���@���H���Xbmodel_3/conv2d_86/Conv2Dh
�
�_ZN13cutlass_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi64ELi256ELi32EEENS4_51Conv2dWgradOutputGradientTileAccessIteratorAnalyticINS_11MatrixShapeILi64ELi32EEENS_6half_tENS_9transform29PitchLinearWarpRakedThreadMapINS_6layout16PitchLinearShapeILi64ELi32EEELi128ENSG_ILi8ELi4EEELi8EEEEENSD_11threadblock25RegularTileAccessIteratorISB_SC_NSF_40ColumnMajorTensorOpMultiplicandCongruousILi16ELi64EEELi1ESJ_Li16EEELNS_4arch14CacheOperation4KindE0ENS4_47Conv2dWgradActivationTileAccessIteratorAnalyticINSA_ILi32ELi256EEESC_NSE_INSG_ILi256ELi32EEELi128ESI_Li8EEEEENSM_ISU_SC_NSF_37RowMajorTensorOpMultiplicandCongruousILi16ELi64EEELi0ESW_Li16EEELSS_0ENS6_11threadblock9MmaPolicyINS6_4warp11MmaTensorOpINS7_ILi64ELi64ELi32EEESC_SO_SC_SZ_fNSF_8RowMajorENS13_17MmaTensorOpPolicyINSQ_3MmaINS7_ILi16ELi8ELi16EEELi32ESC_S16_SC_NSF_11ColumnMajorEfS16_NSQ_13OpMultiplyAddEEENSA_ILi1ELi1EEEEELi1ELb0EbEENSA_ILi0ELi0EEES1G_Li1EEELi4EbEENS_8epilogue11threadblock8EpilogueIS8_S1F_Li1ENS1K_22PredicatedTileIteratorINS1K_26OutputTileOptimalThreadMapINS1K_15OutputTileShapeILi256ELi8ELi1ELi1ELi1EEENS1O_ILi1ELi8ELi1ELi1ELi8EEELi128ELi4ELi32EEEfEENS1J_4warp24FragmentIteratorTensorOpIS15_S19_fNS_5ArrayIfLi4ELb1EEES16_EENS1T_20TileIteratorTensorOpIS15_S19_fS16_EENS1K_18SharedLoadIteratorINS1R_18CompactedThreadMapEfLi16EEENS1J_6thread17LinearCombinationIfLi4EffLNS_15FloatRoundStyleE2EEENSA_ILi0ELi8EEEEENS11_30GemmIdentityThreadblockSwizzleILi4EEELNS1_8OperatorE2ENS1_17Conv2dProblemSizeEEEEEvNT_6ParamsE� ��*�2
8��@��H��Xb;gradient_tape/model_3/conv2d_86/Conv2D/Conv2DBackpropFilterh
�
�void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3>(cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3::Params)� ��*�2�8���@���H���Xbmodel_3/conv2d_94/Conv2Dh
�
�void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride>(cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride::Params)� ��*�2�8�ǻ@�ǻH�ǻXb:gradient_tape/model_3/conv2d_86/Conv2D/Conv2DBackpropInputh
�
�void xmma_cudnn::gemm::kernel<xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, xmma_cudnn::Col, 256, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 3> >(xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, xmma_cudnn::Col, 256, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 3>::Params)� ��*�2Z8��@��H��PXb;gradient_tape/model_3/conv2d_94/Conv2D/Conv2DBackpropFilterh
�
�void cudnn::ops::pooling_bw_kernel_max<__half, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, false>(cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor) � *�2� 8�@�H�b:gradient_tape/model_3/max_pooling2d_13/MaxPool/MaxPoolGradhu  �B
O
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2��8���@���H���bmul_58huZU�B
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2��8�@�H�b,gradient_tape/dense_9/kernel/Regularizer/MulhuZU�B
m
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*�2��8�@�H�b!dense_9/kernel/Regularizer/SquarehuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8�@�H�b.gradient_tape/model_3/dense_9/MatMul/Cast/CasthuZU�B
�
�void cudnn::ops::pooling_bw_kernel_max<__half, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, false>(cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor) � *�2   8�צ@�צH�צb:gradient_tape/model_3/max_pooling2d_12/MaxPool/MaxPoolGradhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8�@�H�bmodel_3/dense_9/MatMul/CasthuZU�B
�
�void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride>(cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride::Params)� ��*�2�8�@�H�Xb:gradient_tape/model_3/conv2d_94/Conv2D/Conv2DBackpropInputh
�
|sm80_xmma_fprop_implicit_gemm_f16f16_f16f32_f32_nhwckrsc_nhwc_tilesize256x64x32_stage3_warpsize4x1x1_g1_tensor16x8x16_kernel� ��*�2 8��@��H��PXbmodel_3/conv2d_95/Conv2Dh
�
�_ZN13cutlass_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi64ELi256ELi32EEENS4_51Conv2dWgradOutputGradientTileAccessIteratorAnalyticINS_11MatrixShapeILi64ELi32EEENS_6half_tENS_9transform29PitchLinearWarpRakedThreadMapINS_6layout16PitchLinearShapeILi64ELi32EEELi128ENSG_ILi8ELi4EEELi8EEEEENSD_11threadblock25RegularTileAccessIteratorISB_SC_NSF_40ColumnMajorTensorOpMultiplicandCongruousILi16ELi64EEELi1ESJ_Li16EEELNS_4arch14CacheOperation4KindE0ENS4_47Conv2dWgradActivationTileAccessIteratorAnalyticINSA_ILi32ELi256EEESC_NSE_INSG_ILi256ELi32EEELi128ESI_Li8EEEEENSM_ISU_SC_NSF_37RowMajorTensorOpMultiplicandCongruousILi16ELi64EEELi0ESW_Li16EEELSS_0ENS6_11threadblock9MmaPolicyINS6_4warp11MmaTensorOpINS7_ILi64ELi64ELi32EEESC_SO_SC_SZ_fNSF_8RowMajorENS13_17MmaTensorOpPolicyINSQ_3MmaINS7_ILi16ELi8ELi16EEELi32ESC_S16_SC_NSF_11ColumnMajorEfS16_NSQ_13OpMultiplyAddEEENSA_ILi1ELi1EEEEELi1ELb0EbEENSA_ILi0ELi0EEES1G_Li1EEELi4EbEENS_8epilogue11threadblock8EpilogueIS8_S1F_Li1ENS1K_22PredicatedTileIteratorINS1K_26OutputTileOptimalThreadMapINS1K_15OutputTileShapeILi256ELi8ELi1ELi1ELi1EEENS1O_ILi1ELi8ELi1ELi1ELi8EEELi128ELi4ELi32EEEfEENS1J_4warp24FragmentIteratorTensorOpIS15_S19_fNS_5ArrayIfLi4ELb1EEES16_EENS1T_20TileIteratorTensorOpIS15_S19_fS16_EENS1K_18SharedLoadIteratorINS1R_18CompactedThreadMapEfLi16EEENS1J_6thread17LinearCombinationIfLi4EffLNS_15FloatRoundStyleE2EEENSA_ILi0ELi8EEEEENS11_30GemmIdentityThreadblockSwizzleILi4EEELNS1_8OperatorE2ENS1_17Conv2dProblemSizeEEEEEvNT_6ParamsE� ��*�2 8�߄@�߄H�߄Xb;gradient_tape/model_3/conv2d_95/Conv2D/Conv2DBackpropFilterh
�
�void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride>(cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride::Params)� ��*�2�8��k@��kH��kXb:gradient_tape/model_3/conv2d_95/Conv2D/Conv2DBackpropInputh
U
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�2��8��i@��iH��ibIsFinite_56hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�2{8��i@��iH��ib-gradient_tape/dense_9/kernel/Regularizer/TilehuZU�B
�
�void xmma_cudnn::gemm::kernel<xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 32, 2, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 32, 2, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 32, 2, 2, 1, 1>, 16, xmma_cudnn::Col, 128, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 32, 2, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 32, 2, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 4> >(xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 32, 2, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 32, 2, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 32, 2, 2, 1, 1>, 16, xmma_cudnn::Col, 128, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 32, 2, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 32, 2, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 4>::Params)� ��*�2	8��f@��fH��fPXb;gradient_tape/model_3/conv2d_80/Conv2D/Conv2DBackpropFilterh
�
�void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride>(cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride::Params)� ��*�2�8��c@��cH��cXb:gradient_tape/model_3/conv2d_88/Conv2D/Conv2DBackpropInputh
�
�_ZN13cutlass_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi64ELi256ELi32EEENS4_51Conv2dWgradOutputGradientTileAccessIteratorAnalyticINS_11MatrixShapeILi64ELi32EEENS_6half_tENS_9transform29PitchLinearWarpRakedThreadMapINS_6layout16PitchLinearShapeILi64ELi32EEELi128ENSG_ILi8ELi4EEELi8EEEEENSD_11threadblock25RegularTileAccessIteratorISB_SC_NSF_40ColumnMajorTensorOpMultiplicandCongruousILi16ELi64EEELi1ESJ_Li16EEELNS_4arch14CacheOperation4KindE0ENS4_47Conv2dWgradActivationTileAccessIteratorAnalyticINSA_ILi32ELi256EEESC_NSE_INSG_ILi256ELi32EEELi128ESI_Li8EEEEENSM_ISU_SC_NSF_37RowMajorTensorOpMultiplicandCongruousILi16ELi64EEELi0ESW_Li16EEELSS_0ENS6_11threadblock9MmaPolicyINS6_4warp11MmaTensorOpINS7_ILi64ELi64ELi32EEESC_SO_SC_SZ_fNSF_8RowMajorENS13_17MmaTensorOpPolicyINSQ_3MmaINS7_ILi16ELi8ELi16EEELi32ESC_S16_SC_NSF_11ColumnMajorEfS16_NSQ_13OpMultiplyAddEEENSA_ILi1ELi1EEEEELi1ELb0EbEENSA_ILi0ELi0EEES1G_Li1EEELi4EbEENS_8epilogue11threadblock8EpilogueIS8_S1F_Li1ENS1K_22PredicatedTileIteratorINS1K_26OutputTileOptimalThreadMapINS1K_15OutputTileShapeILi256ELi8ELi1ELi1ELi1EEENS1O_ILi1ELi8ELi1ELi1ELi8EEELi128ELi4ELi32EEEfEENS1J_4warp24FragmentIteratorTensorOpIS15_S19_fNS_5ArrayIfLi4ELb1EEES16_EENS1T_20TileIteratorTensorOpIS15_S19_fS16_EENS1K_18SharedLoadIteratorINS1R_18CompactedThreadMapEfLi16EEENS1J_6thread17LinearCombinationIfLi4EffLNS_15FloatRoundStyleE2EEENSA_ILi0ELi8EEEEENS11_30GemmIdentityThreadblockSwizzleILi4EEELNS1_8OperatorE2ENS1_17Conv2dProblemSizeEEEEEvNT_6ParamsE� ��*�2
8��a@��aH��aXb;gradient_tape/model_3/conv2d_88/Conv2D/Conv2DBackpropFilterh
�
�void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3>(cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3::Params)� ��*�2�8��`@��`H��`Xbmodel_3/conv2d_88/Conv2Dh
�
�void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3>(cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3::Params)� ��*�2�8��^@��^H��^Xbmodel_3/conv2d_80/Conv2Dh
�
�void xmma_cudnn::gemm::kernel<xmma_cudnn::implicit_gemm::dgrad::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_a_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 16, false, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 16, xmma_cudnn::Row, 32, 256> >, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_c_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, 16, xmma_cudnn::Fragment_c<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, false>, false>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 3> >(xmma_cudnn::implicit_gemm::dgrad::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_a_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 16, false, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 16, xmma_cudnn::Row, 32, 256> >, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_c_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, 16, xmma_cudnn::Fragment_c<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, false>, false>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 3>::Params)� ��*�2�8��\@��\H��\PXb:gradient_tape/model_3/conv2d_80/Conv2D/Conv2DBackpropInputh
�
�void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*�2�8��R@��RH��Rbdense_9/kernel/Regularizer/Sumhu  �B
�
�void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride>(cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride::Params)� ��*�2@8��P@��PH��PXb:gradient_tape/model_3/conv2d_92/Conv2D/Conv2DBackpropInputh
�
�void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3>(cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3::Params)� ��*�2�8��M@��MH��MXbmodel_3/conv2d_85/Conv2Dh
�
�_ZN13cutlass_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi64ELi256ELi32EEENS4_51Conv2dWgradOutputGradientTileAccessIteratorAnalyticINS_11MatrixShapeILi64ELi32EEENS_6half_tENS_9transform29PitchLinearWarpRakedThreadMapINS_6layout16PitchLinearShapeILi64ELi32EEELi128ENSG_ILi8ELi4EEELi8EEEEENSD_11threadblock25RegularTileAccessIteratorISB_SC_NSF_40ColumnMajorTensorOpMultiplicandCongruousILi16ELi64EEELi1ESJ_Li16EEELNS_4arch14CacheOperation4KindE0ENS4_47Conv2dWgradActivationTileAccessIteratorAnalyticINSA_ILi32ELi256EEESC_NSE_INSG_ILi256ELi32EEELi128ESI_Li8EEEEENSM_ISU_SC_NSF_37RowMajorTensorOpMultiplicandCongruousILi16ELi64EEELi0ESW_Li16EEELSS_0ENS6_11threadblock9MmaPolicyINS6_4warp11MmaTensorOpINS7_ILi64ELi64ELi32EEESC_SO_SC_SZ_fNSF_8RowMajorENS13_17MmaTensorOpPolicyINSQ_3MmaINS7_ILi16ELi8ELi16EEELi32ESC_S16_SC_NSF_11ColumnMajorEfS16_NSQ_13OpMultiplyAddEEENSA_ILi1ELi1EEEEELi1ELb0EbEENSA_ILi0ELi0EEES1G_Li1EEELi4EbEENS_8epilogue11threadblock8EpilogueIS8_S1F_Li1ENS1K_22PredicatedTileIteratorINS1K_26OutputTileOptimalThreadMapINS1K_15OutputTileShapeILi256ELi8ELi1ELi1ELi1EEENS1O_ILi1ELi8ELi1ELi1ELi8EEELi128ELi4ELi32EEEfEENS1J_4warp24FragmentIteratorTensorOpIS15_S19_fNS_5ArrayIfLi4ELb1EEES16_EENS1T_20TileIteratorTensorOpIS15_S19_fS16_EENS1K_18SharedLoadIteratorINS1R_18CompactedThreadMapEfLi16EEENS1J_6thread17LinearCombinationIfLi4EffLNS_15FloatRoundStyleE2EEENSA_ILi0ELi8EEEEENS11_30GemmIdentityThreadblockSwizzleILi4EEELNS1_8OperatorE2ENS1_17Conv2dProblemSizeEEEEEvNT_6ParamsE� ��*�2
8��I@��IH��IXb;gradient_tape/model_3/conv2d_85/Conv2D/Conv2DBackpropFilterh
�
�_ZN13cutlass_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi64ELi256ELi32EEENS4_51Conv2dWgradOutputGradientTileAccessIteratorAnalyticINS_11MatrixShapeILi64ELi32EEENS_6half_tENS_9transform29PitchLinearWarpRakedThreadMapINS_6layout16PitchLinearShapeILi64ELi32EEELi128ENSG_ILi8ELi4EEELi8EEEEENSD_11threadblock25RegularTileAccessIteratorISB_SC_NSF_40ColumnMajorTensorOpMultiplicandCongruousILi16ELi64EEELi1ESJ_Li16EEELNS_4arch14CacheOperation4KindE0ENS4_47Conv2dWgradActivationTileAccessIteratorAnalyticINSA_ILi32ELi256EEESC_NSE_INSG_ILi256ELi32EEELi128ESI_Li8EEEEENSM_ISU_SC_NSF_37RowMajorTensorOpMultiplicandCongruousILi16ELi64EEELi0ESW_Li16EEELSS_0ENS6_11threadblock9MmaPolicyINS6_4warp11MmaTensorOpINS7_ILi64ELi64ELi32EEESC_SO_SC_SZ_fNSF_8RowMajorENS13_17MmaTensorOpPolicyINSQ_3MmaINS7_ILi16ELi8ELi16EEELi32ESC_S16_SC_NSF_11ColumnMajorEfS16_NSQ_13OpMultiplyAddEEENSA_ILi1ELi1EEEEELi1ELb0EbEENSA_ILi0ELi0EEES1G_Li1EEELi4EbEENS_8epilogue11threadblock8EpilogueIS8_S1F_Li1ENS1K_22PredicatedTileIteratorINS1K_26OutputTileOptimalThreadMapINS1K_15OutputTileShapeILi256ELi8ELi1ELi1ELi1EEENS1O_ILi1ELi8ELi1ELi1ELi8EEELi128ELi4ELi32EEEfEENS1J_4warp24FragmentIteratorTensorOpIS15_S19_fNS_5ArrayIfLi4ELb1EEES16_EENS1T_20TileIteratorTensorOpIS15_S19_fS16_EENS1K_18SharedLoadIteratorINS1R_18CompactedThreadMapEfLi16EEENS1J_6thread17LinearCombinationIfLi4EffLNS_15FloatRoundStyleE2EEENSA_ILi0ELi8EEEEENS11_30GemmIdentityThreadblockSwizzleILi4EEELNS1_8OperatorE2ENS1_17Conv2dProblemSizeEEEEEvNT_6ParamsE� ��*�2	8��H@��HH��HXb;gradient_tape/model_3/conv2d_92/Conv2D/Conv2DBackpropFilterh
�
�void xmma_cudnn::gemm::kernel<xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, xmma_cudnn::Col, 256, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 3> >(xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, xmma_cudnn::Col, 256, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 3>::Params)� ��*�2-8��G@��GH��GPXb;gradient_tape/model_3/conv2d_91/Conv2D/Conv2DBackpropFilterh
�
�void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride>(cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride::Params)� ��*�2�8��E@��EH��EXb:gradient_tape/model_3/conv2d_85/Conv2D/Conv2DBackpropInputh
�
�void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3>(cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3::Params)� ��*�2�8��D@��DH��DXbmodel_3/conv2d_79/Conv2Dh
�
�void xmma_cudnn::gemm::kernel<xmma_cudnn::implicit_gemm::dgrad::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_a_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 16, false, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 16, xmma_cudnn::Row, 32, 256> >, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_c_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, 16, xmma_cudnn::Fragment_c<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, false>, false>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 3> >(xmma_cudnn::implicit_gemm::dgrad::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_a_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 16, false, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 16, xmma_cudnn::Row, 32, 256> >, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_c_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, 16, xmma_cudnn::Fragment_c<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, false>, false>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 3>::Params)� ��*�2�8��C@��CH��CPXb:gradient_tape/model_3/conv2d_79/Conv2D/Conv2DBackpropInputh
�
|sm80_xmma_fprop_implicit_gemm_f16f16_f16f32_f32_nhwckrsc_nhwc_tilesize256x64x32_stage3_warpsize4x1x1_g1_tensor16x8x16_kernel� ��*�2 8��C@��CH��CPXbmodel_3/conv2d_92/Conv2Dh
�
�void xmma_cudnn::gemm::kernel<xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, xmma_cudnn::Col, 256, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 3> >(xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, xmma_cudnn::Col, 256, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 3>::Params)� ��*�2		8��@@��@H��@PXb;gradient_tape/model_3/conv2d_79/Conv2D/Conv2DBackpropFilterh
�
�void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride>(cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride::Params)� ��*�2@8��:@��:H��:Xb:gradient_tape/model_3/conv2d_91/Conv2D/Conv2DBackpropInputh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) �*�2��8��:@��H��bAgradient_tape/model_3/batch_normalization_33/FusedBatchNormGradV3hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) �*�2��8��:@��H��bAgradient_tape/model_3/batch_normalization_32/FusedBatchNormGradV3hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) �*�2��8��:@��H��bAgradient_tape/model_3/batch_normalization_31/FusedBatchNormGradV3hu  �B
�
�void cudnn::bn_bw_1C11_kernel_new<__half, float, float2, 512, true, 1>(float, float, float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, float const*, float*, float*, float const*, float const*, float)(�*�2�8��8@��8H��8bAgradient_tape/model_3/batch_normalization_31/FusedBatchNormGradV3hu  �B
�
�void cudnn::ops::pooling_bw_kernel_max<__half, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, false>(cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor) � *�2� 8��6@��6H��6b:gradient_tape/model_3/max_pooling2d_14/MaxPool/MaxPoolGradhu  �B
�
�void cudnn::bn_bw_1C11_kernel_new<__half, float, float2, 512, true, 1>(float, float, float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, float const*, float*, float*, float const*, float const*, float)(�*�2�8��6@��6H��6bAgradient_tape/model_3/batch_normalization_32/FusedBatchNormGradV3hu  �B
�
�void cudnn::bn_bw_1C11_kernel_new<__half, float, float2, 512, true, 1>(float, float, float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, float const*, float*, float*, float const*, float const*, float)(�*�2�8��5@��5H��5bAgradient_tape/model_3/batch_normalization_33/FusedBatchNormGradV3hu  �B
�
�void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3>(cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3::Params)� ��*�2�8��0@��0H��0Xbmodel_3/conv2d_91/Conv2Dh
}
-ampere_fp16_s1688gemm_fp16_256x64_ldg8_f2f_nt���*�2�8��-@��-H��-b&gradient_tape/model_3/dense_9/MatMul_1hugU�A
z
:ampere_fp16_s16816gemm_fp16_128x64_ldg8_f2f_stages_64x4_nn��� ��*�28��-@��-H��-Xbmodel_3/dense_9/MatMulh
�
:ampere_fp16_s16816gemm_fp16_128x64_ldg8_f2f_stages_64x4_tn��� ��*�2�	8��,@��,H��,Xb$gradient_tape/model_3/dense_9/MatMulh
�
�void cudnn::ops::pooling_bw_kernel_max<__half, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, false>(cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor) � *�2� 8��)@��)H��)b:gradient_tape/model_3/max_pooling2d_15/MaxPool/MaxPoolGradhu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) �*�2��8��'@��H��b/model_3/batch_normalization_32/FusedBatchNormV3hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) �*�2��8��&@��H��b/model_3/batch_normalization_33/FusedBatchNormV3hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) �*�2��8��&@��H��b/model_3/batch_normalization_31/FusedBatchNormV3hu  �B
�

�
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8��&@��&H��&bAdam/gradients/AddN_8huZU�B
�
�void cudnn::bn_fw_tr_1C11_kernel_NCHW<__half, float, 512, true, 1>(cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(�*�2�8�� @�� H�� b/model_3/batch_normalization_33/FusedBatchNormV3hu  �B
�
�void cudnn::bn_fw_tr_1C11_kernel_NCHW<__half, float, 512, true, 1>(cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(�*�2�8�� @�� H�� b/model_3/batch_normalization_32/FusedBatchNormV3hu  �B
�
�void convolve_common_engine_float_NHWC<__half, __half, 128, 5, 5, 3, 3, 3, true, false, false, false, false>(int, int, int, __half const*, __half const*, int, __half*, conv_kernel_common_params, unsigned long long, unsigned long, float, float, bool, __half const*, __half const*, bool)P�*2��8��@��H��Xbmodel_3/conv2d_77/Conv2Dhu  HB
�
�_ZN13cutlass_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi64ELi256ELi32EEENS4_51Conv2dWgradOutputGradientTileAccessIteratorAnalyticINS_11MatrixShapeILi64ELi32EEENS_6half_tENS_9transform29PitchLinearWarpRakedThreadMapINS_6layout16PitchLinearShapeILi64ELi32EEELi128ENSG_ILi8ELi4EEELi8EEEEENSD_11threadblock25RegularTileAccessIteratorISB_SC_NSF_40ColumnMajorTensorOpMultiplicandCongruousILi16ELi64EEELi1ESJ_Li16EEELNS_4arch14CacheOperation4KindE0ENS4_47Conv2dWgradActivationTileAccessIteratorAnalyticINSA_ILi32ELi256EEESC_NSE_INSG_ILi256ELi32EEELi128ESI_Li8EEEEENSM_ISU_SC_NSF_37RowMajorTensorOpMultiplicandCongruousILi16ELi64EEELi0ESW_Li16EEELSS_0ENS6_11threadblock9MmaPolicyINS6_4warp11MmaTensorOpINS7_ILi64ELi64ELi32EEESC_SO_SC_SZ_fNSF_8RowMajorENS13_17MmaTensorOpPolicyINSQ_3MmaINS7_ILi16ELi8ELi16EEELi32ESC_S16_SC_NSF_11ColumnMajorEfS16_NSQ_13OpMultiplyAddEEENSA_ILi1ELi1EEEEELi1ELb0EbEENSA_ILi0ELi0EEES1G_Li1EEELi4EbEENS_8epilogue11threadblock8EpilogueIS8_S1F_Li1ENS1K_22PredicatedTileIteratorINS1K_26OutputTileOptimalThreadMapINS1K_15OutputTileShapeILi256ELi8ELi1ELi1ELi1EEENS1O_ILi1ELi8ELi1ELi1ELi8EEELi128ELi4ELi32EEEfEENS1J_4warp24FragmentIteratorTensorOpIS15_S19_fNS_5ArrayIfLi4ELb1EEES16_EENS1T_20TileIteratorTensorOpIS15_S19_fS16_EENS1K_18SharedLoadIteratorINS1R_18CompactedThreadMapEfLi16EEENS1J_6thread17LinearCombinationIfLi4EffLNS_15FloatRoundStyleE2EEENSA_ILi0ELi8EEEEENS11_30GemmIdentityThreadblockSwizzleILi4EEELNS1_8OperatorE2ENS1_17Conv2dProblemSizeEEEEEvNT_6ParamsE� ��*�2)8��@��H��Xb;gradient_tape/model_3/conv2d_77/Conv2D/Conv2DBackpropFilterh
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*�2{8��@��H��bmodel_3/concatenate_25/concathuZU�B
�
�void cudnn::bn_fw_tr_1C11_kernel_NCHW<__half, float, 512, true, 1>(cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(�*�2�8��@��H��b/model_3/batch_normalization_31/FusedBatchNormV3hu  �B
�
htensorflow::functor::ReluGradHalfKernelVector(Eigen::half const*, Eigen::half const*, Eigen::half*, int)*�2��8��@��H��b,gradient_tape/model_3/activation_35/ReluGradhu  �B
�
htensorflow::functor::ReluGradHalfKernelVector(Eigen::half const*, Eigen::half const*, Eigen::half*, int)*�2��8��@��H��b,gradient_tape/model_3/activation_34/ReluGradhu  �B
�
htensorflow::functor::ReluGradHalfKernelVector(Eigen::half const*, Eigen::half const*, Eigen::half*, int)*�2��8��@��H��b,gradient_tape/model_3/activation_36/ReluGradhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*�2{8��@��H��bmodel_3/concatenate_27/concathuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*�2{8��@��H��bmodel_3/concatenate_26/concathuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�2R8��@��H��b<cond_1/then/_10/cond_1/Adam/Adam/update_50/ResourceApplyAdamhuZU�B
�
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_tn��� ��*�2�8��@��H��Xb:gradient_tape/model_3/conv2d_81/Conv2D/Conv2DBackpropInputh
�
�void cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *�2�8��@��H��bAll_56hu  �B
}
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_nn��� ��*�2@8��@��H��Xbmodel_3/conv2d_93/Conv2Dh
�	
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*�2{8��@��H��bmodel_3/activation_35/ReluhuZU�B
�	
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*�2{8��@��H��bmodel_3/activation_34/ReluhuZU�B
�	
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*�2{8��@��H��bmodel_3/activation_36/ReluhuZU�B
�
�void cutlass_cudnn::Kernel<cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4> >(cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4>::Params)6* 2 28��@��H��Xb;gradient_tape/model_3/conv2d_83/Conv2D/Conv2DBackpropFilterhu  �B
�
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_tn��� ��*�2
@8��@��H��Xb:gradient_tape/model_3/conv2d_93/Conv2D/Conv2DBackpropInputh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) �*�2��8��@��H��bAgradient_tape/model_3/batch_normalization_34/FusedBatchNormGradV3hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) �*�2��8��@��H��bAgradient_tape/model_3/batch_normalization_35/FusedBatchNormGradV3hu  �B
�
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_64x3_nt��� ��*�2
8��@��H��Xb;gradient_tape/model_3/conv2d_93/Conv2D/Conv2DBackpropFilterh
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�2R8��@��H��b<cond_1/then/_10/cond_1/Adam/Adam/update_52/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) �*�2��8��@��H��bAgradient_tape/model_3/batch_normalization_37/FusedBatchNormGradV3hu  �B
~
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_nn��� ��*�2�8��@��H��Xbmodel_3/conv2d_81/Conv2Dh
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�2{8��@��H��b,gradient_tape/model_3/concatenate_25/Slice_2huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�2{8��@��H��b,gradient_tape/model_3/concatenate_26/Slice_1huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�2{8��@��H��b,gradient_tape/model_3/concatenate_27/Slice_1huZU�B
�
�void cudnn::bn_bw_1C11_kernel_new<__half, float, float2, 128, true, 1>(float, float, float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, float const*, float*, float*, float const*, float const*, float)(�*�2�8��@��H��bAgradient_tape/model_3/batch_normalization_34/FusedBatchNormGradV3hu  �B
�
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_64x3_nt��� ��*�28��@��H��Xb;gradient_tape/model_3/conv2d_81/Conv2D/Conv2DBackpropFilterh
�
�void cudnn::bn_bw_1C11_kernel_new<__half, float, float2, 128, true, 1>(float, float, float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, float const*, float*, float*, float const*, float const*, float)(�*�2�8��@��H��bAgradient_tape/model_3/batch_normalization_35/FusedBatchNormGradV3hu  �B
�
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*�2R8��@��H��bmodel_3/conv2d_82/BiasAddhuZU�B
�
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*�2R8��@��H��bmodel_3/conv2d_77/BiasAddhuZU�B
�
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*�2R8��@��H��bmodel_3/conv2d_79/BiasAddhuZU�B
�
�void pooling_fw_kernel_max_nhwc<__half, float, 0, (cudnnNanPropagation_t)0, false>(PoolingParams, __half*, __half const*, float, float, int)*�2��8��@��H��b model_3/max_pooling2d_13/MaxPoolhu  �B
�
�void pooling_fw_kernel_max_nhwc<__half, float, 0, (cudnnNanPropagation_t)0, false>(PoolingParams, __half*, __half const*, float, float, int)*�2��8��@��H��b model_3/max_pooling2d_12/MaxPoolhu  �B
�
�void cutlass_cudnn::Kernel<cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4> >(cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4>::Params)6* 2@}8��@��H��Xb;gradient_tape/model_3/conv2d_95/Conv2D/Conv2DBackpropFilterhu  �B
�
�_ZN13cutlass_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi64ELi64ELi32EEENS4_51Conv2dWgradOutputGradientTileAccessIteratorAnalyticINS_11MatrixShapeILi64ELi32EEENS_6half_tENS_9transform29PitchLinearWarpRakedThreadMapINS_6layout16PitchLinearShapeILi64ELi32EEELi128ENSG_ILi8ELi4EEELi8EEEEENSD_11threadblock25RegularTileAccessIteratorISB_SC_NSF_40ColumnMajorTensorOpMultiplicandCongruousILi16ELi64EEELi1ESJ_Li16EEELNS_4arch14CacheOperation4KindE0ENS4_47Conv2dWgradActivationTileAccessIteratorAnalyticINSA_ILi32ELi64EEESC_SJ_EENSM_ISU_SC_NSF_37RowMajorTensorOpMultiplicandCongruousILi16ELi64EEELi0ESJ_Li16EEELSS_0ENS6_11threadblock9MmaPolicyINS6_4warp11MmaTensorOpINS7_ILi32ELi32ELi32EEESC_SO_SC_SX_fNSF_8RowMajorENS11_17MmaTensorOpPolicyINSQ_3MmaINS7_ILi16ELi8ELi16EEELi32ESC_S14_SC_NSF_11ColumnMajorEfS14_NSQ_13OpMultiplyAddEEENSA_ILi1ELi1EEEEELi1ELb0EbEENSA_ILi0ELi0EEES1E_Li1EEELi10EbEENS_8epilogue11threadblock8EpilogueIS8_S1D_Li1ENS1I_22PredicatedTileIteratorINS1I_26OutputTileOptimalThreadMapINS1I_15OutputTileShapeILi64ELi8ELi2ELi1ELi1EEENS1M_ILi1ELi4ELi1ELi1ELi4EEELi128ELi4ELi32EEEfEENS1H_4warp24FragmentIteratorTensorOpIS13_S17_fNS_5ArrayIfLi4ELb1EEES14_EENS1R_20TileIteratorTensorOpIS13_S17_fS14_EENS1I_18SharedLoadIteratorINS1P_18CompactedThreadMapEfLi16EEENS1H_6thread17LinearCombinationIfLi4EffLNS_15FloatRoundStyleE2EEENSA_ILi0ELi8EEEEENSZ_30GemmIdentityThreadblockSwizzleILi4EEELNS1_8OperatorE2ENS1_17Conv2dProblemSizeEEEEEvNT_6ParamsEh ��*�2)8��@��H��Xb;gradient_tape/model_3/conv2d_76/Conv2D/Conv2DBackpropFilterh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) �*�2��8��@��H��b/model_3/batch_normalization_34/FusedBatchNormV3hu  �B
�
�void cutlass_cudnn::Kernel<cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4> >(cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4>::Params)6* 2@?8��@��H��Xb;gradient_tape/model_3/conv2d_89/Conv2D/Conv2DBackpropFilterhu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) �*�2��8��@��H��b/model_3/batch_normalization_35/FusedBatchNormV3hu  �B
�

�
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8��@��H��bAdam/gradients/AddN_6huZU�B
�
�void cutlass_cudnn::Kernel<cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4> >(cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4>::Params)6* 2@?8��@��H��Xb;gradient_tape/model_3/conv2d_92/Conv2D/Conv2DBackpropFilterhu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) �*�2��8��@��H��b/model_3/batch_normalization_37/FusedBatchNormV3hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8��
@��
H��
bAdam/gradients/AddN_29huZU�B
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��
@��
H��
b0gradient_tape/conv2d_94/kernel/Regularizer/Mul_1huZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*�2R8��
@��
H��
Xb;gradient_tape/model_3/conv2d_94/Conv2D/Conv2DBackpropFilterhuZU�B
�
�void cutlass_cudnn::Kernel<cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4> >(cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4>::Params)6* 2@28��
@��
H��
Xb;gradient_tape/model_3/conv2d_86/Conv2D/Conv2DBackpropFilterhu  �B
�
�void tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)�*  2 8��
@��
H��
b3gradient_tape/model_3/conv2d_77/BiasAdd/BiasAddGradhuZU�B
�
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_64x3_nt��� ��*�28��
@��
H��
Xb;gradient_tape/model_3/conv2d_84/Conv2D/Conv2DBackpropFilterh
�
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_64x3_nt��� ��*�28��
@��
H��
Xb;gradient_tape/model_3/conv2d_87/Conv2D/Conv2DBackpropFilterh
�
�void tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)�*  2 8��	@��	H��	b3gradient_tape/model_3/conv2d_82/BiasAdd/BiasAddGradhuZU�B
�
�void tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)�*  2 8��	@��	H��	b3gradient_tape/model_3/conv2d_76/BiasAdd/BiasAddGradhuZU�B
�
�void tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)�*  2 8��	@��	H��	b3gradient_tape/model_3/conv2d_79/BiasAdd/BiasAddGradhuZU�B
�

�
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8��	@��	H��	bAdam/gradients/AddN_7huZU�B
�
�void tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)�*  2 8��	@��	H��	b3gradient_tape/model_3/conv2d_75/BiasAdd/BiasAddGradhuZU�B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) �*�2�P8��	@��H��bAgradient_tape/model_3/batch_normalization_36/FusedBatchNormGradV3hu  �B
�

�
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8��	@��	H��	bAdam/gradients/AddN_9huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*�2{8��	@��H��bmodel_3/concatenate_28/concathuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*�2{8��	@��H��bmodel_3/concatenate_29/concathuZU�B
�
Nvoid cudnn::ops::scalePackedTensor_kernel<__half, float>(long, __half*, float)*�2��8��	@��	H��	b:gradient_tape/model_3/max_pooling2d_12/MaxPool/MaxPoolGradhu  �B
�
Nvoid cudnn::ops::scalePackedTensor_kernel<__half, float>(long, __half*, float)*�2��8��	@��	H��	b:gradient_tape/model_3/max_pooling2d_13/MaxPool/MaxPoolGradhu  �B
�
htensorflow::functor::ReluGradHalfKernelVector(Eigen::half const*, Eigen::half const*, Eigen::half*, int)*�2�(8��	@��	H��	b,gradient_tape/model_3/activation_37/ReluGradhu  �B
�
�void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*�2R8��	@��	H��	Xbmodel_3/conv2d_94/Conv2DhuZU�B
�
htensorflow::functor::ReluGradHalfKernelVector(Eigen::half const*, Eigen::half const*, Eigen::half*, int)*�2�(8��	@��	H��	b,gradient_tape/model_3/activation_38/ReluGradhu  �B
�
�void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*�2R8��	@��	H��	Xb:gradient_tape/model_3/conv2d_94/Conv2D/Conv2DBackpropInputhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�2R8��@��H��b<cond_1/then/_10/cond_1/Adam/Adam/update_60/ResourceApplyAdamhuZU�B
�
�void cudnn::bn_bw_1C11_singleread_specialized<__half2, 512, 1, 2, 7>(float, float, float, float, cudnnTensorStruct, __half2 const*, cudnnTensorStruct, __half2 const*, cudnnTensorStruct, __half2*, float const*, float*, float*, float const*, float const*, float, cudnn::reduced_divisor, int, cudnn::reduced_divisor, cudnn::bnBwPersistentState*, int, float, float, float, int, float, cudnnStatus_t*, bool)8� � *�2�8��@��H��bAgradient_tape/model_3/batch_normalization_37/FusedBatchNormGradV3huZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�2R8��@��H��b<cond_1/then/_10/cond_1/Adam/Adam/update_44/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�2R8��@��H��b<cond_1/then/_10/cond_1/Adam/Adam/update_36/ResourceApplyAdamhuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *�2{8��@��H��b/gradient_tape/conv2d_94/kernel/Regularizer/TilehuZU�B
�
�void convolve_common_engine_float_NHWC<__half, __half, 128, 5, 5, 3, 3, 3, true, false, false, false, false>(int, int, int, __half const*, __half const*, int, __half*, conv_kernel_common_params, unsigned long long, unsigned long, float, float, bool, __half const*, __half const*, bool)P�*2��8��@��H��Xbmodel_3/conv2d_76/Conv2Dhu  HB
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*�2{8��@��H��bmodel_3/concatenate_31/concathuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�2{8��@��H��b*gradient_tape/model_3/concatenate_27/SlicehuZU�B
�
htensorflow::functor::ReluGradHalfKernelVector(Eigen::half const*, Eigen::half const*, Eigen::half*, int)*�2�$8��@��H��b,gradient_tape/model_3/activation_40/ReluGradhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�2{8��@��H��b*gradient_tape/model_3/concatenate_25/SlicehuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�2{8��@��H��b,gradient_tape/model_3/concatenate_25/Slice_1huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�2{8��@��H��b,gradient_tape/model_3/concatenate_27/Slice_2huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�2{8��@��H��b*gradient_tape/model_3/concatenate_26/SlicehuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�2{8��@��H��b,gradient_tape/model_3/concatenate_26/Slice_2huZU�B
�
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_tn��� ��*�2�8��@��H��Xb:gradient_tape/model_3/conv2d_87/Conv2D/Conv2DBackpropInputh
c
Mul_GPU_DT_HALF_DT_HALF_kernel*�2� 8��@��H��b model_3/dropout_18/dropout/Mul_1huZU�B
q
Mul_GPU_DT_HALF_DT_HALF_kernel*�2� 8��@��H��b.gradient_tape/model_3/dropout_19/dropout/Mul_1huZU�B
q
Mul_GPU_DT_HALF_DT_HALF_kernel*�2� 8��@��H��b.gradient_tape/model_3/dropout_18/dropout/Mul_1huZU�B
c
Mul_GPU_DT_HALF_DT_HALF_kernel*�2� 8��@��H��b model_3/dropout_19/dropout/Mul_1huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8��@��H��bAdam/gradients/AddN_30huZU�B
�
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*�2R8��@��H��bmodel_3/conv2d_80/BiasAddhuZU�B
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��b0gradient_tape/conv2d_95/kernel/Regularizer/Mul_1huZU�B
�
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*�2R8��@��H��bmodel_3/conv2d_76/BiasAddhuZU�B
�
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*�2R8��@��H��bmodel_3/conv2d_83/BiasAddhuZU�B
�
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*�2R8��@��H��bmodel_3/conv2d_78/BiasAddhuZU�B
�
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*�2R8��@��H��bmodel_3/conv2d_75/BiasAddhuZU�B
�
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*�2R8��@��H��bmodel_3/conv2d_81/BiasAddhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�2R8��@��H��b<cond_1/then/_10/cond_1/Adam/Adam/update_28/ResourceApplyAdamhuZU�B
K
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��bmul_52huZU�B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��b.gradient_tape/conv2d_94/kernel/Regularizer/MulhuZU�B
k
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��b#conv2d_94/kernel/Regularizer/SquarehuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8��@��H��b0gradient_tape/model_3/conv2d_94/Conv2D/Cast/CasthuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*�2R8��@��H��Xbmodel_3/conv2d_95/Conv2DhuZU�B
�
�void cudnn::bn_fw_tr_1C11_singleread_specialized<__half2, 512, 1, 2, 20>(cudnnTensorStruct, __half2 const*, cudnnTensorStruct, __half2*, float const*, float const*, float, float, float*, float*, float*, float*, float, float, cudnn::reduced_divisor, int, cudnn::reduced_divisor, cudnn::bnFwPersistentState*, int, float, float, float, int, float, float, cudnnStatus_t*, bool)@� ��*�2�8��@��H��b/model_3/batch_normalization_35/FusedBatchNormV3huZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*�2R8��@��H��Xb:gradient_tape/model_3/conv2d_95/Conv2D/Conv2DBackpropInputhuZU�B
~
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_nn��� ��*�2�8��@��H��Xbmodel_3/conv2d_87/Conv2Dh
�
�void cudnn::bn_fw_tr_1C11_singleread_specialized<__half2, 512, 1, 2, 20>(cudnnTensorStruct, __half2 const*, cudnnTensorStruct, __half2*, float const*, float const*, float, float, float*, float*, float*, float*, float, float, cudnn::reduced_divisor, int, cudnn::reduced_divisor, cudnn::bnFwPersistentState*, int, float, float, float, int, float, float, cudnnStatus_t*, bool)@� ��*�2�8��@��H��b/model_3/batch_normalization_34/FusedBatchNormV3huZU�B
�

�
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8��@��H��bAdam/gradients/AddN_4huZU�B
�
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_tn��� ��*�2�8��@��H��Xb:gradient_tape/model_3/conv2d_84/Conv2D/Conv2DBackpropInputh
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�2R8��@��H��b<cond_1/then/_10/cond_1/Adam/Adam/update_42/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) �*�2�P8��@��H��b/model_3/batch_normalization_36/FusedBatchNormV3hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8��@��H��bmodel_3/dropout_19/dropout/CasthuZU�B
}
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_nn��� ��*�2@8��@��H��Xbmodel_3/conv2d_90/Conv2Dh
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8��@��H��bmodel_3/dropout_18/dropout/CasthuZU�B
�	
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*�2{8��@��H��bmodel_3/activation_37/ReluhuZU�B
�	
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*�2{8��@��H��bmodel_3/activation_38/ReluhuZU�B
�
�void cudnn::bn_fw_tr_1C11_singleread_specialized<__half2, 512, 1, 2, 0>(cudnnTensorStruct, __half2 const*, cudnnTensorStruct, __half2*, float const*, float const*, float, float, float*, float*, float*, float*, float, float, cudnn::reduced_divisor, int, cudnn::reduced_divisor, cudnn::bnFwPersistentState*, int, float, float, float, int, float, float, cudnnStatus_t*, bool)(� ��*�2�8��@��H��b/model_3/batch_normalization_37/FusedBatchNormV3hu  �B
�
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_nt��� ��*�28��@��H��Xb;gradient_tape/model_3/conv2d_90/Conv2D/Conv2DBackpropFilterh
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8��@��H��bmodel_3/conv2d_94/Conv2D/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *�2{8��@��H��b/gradient_tape/conv2d_95/kernel/Regularizer/TilehuZU�B
�
1ampere_s16816gemm_fp16_128x64_ldg8_stages_64x4_nt��� ��*�28��@��H��Xb;gradient_tape/model_3/conv2d_78/Conv2D/Conv2DBackpropFilterh
�	
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*�2{8��@��H��bmodel_3/activation_40/ReluhuZU�B
�
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_tn��� ��*�2@8��@��H��Xb:gradient_tape/model_3/conv2d_90/Conv2D/Conv2DBackpropInputh
�
�void tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)�*  2 8��@��H��b3gradient_tape/model_3/conv2d_80/BiasAdd/BiasAddGradhuZU�B
r
.ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_nn���*�2�8��@��H��Xbmodel_3/conv2d_78/Conv2DhugU�A
�
.ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_tn���*�2�8��@��H��Xb:gradient_tape/model_3/conv2d_78/Conv2D/Conv2DBackpropInputhugU�A
�
�void tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)�*  2 8��@��H��b3gradient_tape/model_3/conv2d_81/BiasAdd/BiasAddGradhuZU�B
~
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_nn��� ��*�2�8��@��H��Xbmodel_3/conv2d_84/Conv2Dh
�
�void tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)�*  2 8��@��H��b3gradient_tape/model_3/conv2d_78/BiasAdd/BiasAddGradhuZU�B
�
�void tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)�*  2 8��@��H��b3gradient_tape/model_3/conv2d_83/BiasAdd/BiasAddGradhuZU�B
�
�void tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)�*  2 8��@��H��b3gradient_tape/model_3/conv2d_93/BiasAdd/BiasAddGradhuZU�B
�
�void tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)�*  28��@��H��b3gradient_tape/model_3/conv2d_91/BiasAdd/BiasAddGradhuZU�B
�
�void tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)�*  2 8��@��H��b3gradient_tape/model_3/conv2d_94/BiasAdd/BiasAddGradhuZU�B
o
Mul_GPU_DT_HALF_DT_HALF_kernel*�2� 8��@��H��b,gradient_tape/model_3/dropout_19/dropout/MulhuZU�B
o
Mul_GPU_DT_HALF_DT_HALF_kernel*�2� 8��@��H��b,gradient_tape/model_3/dropout_18/dropout/MulhuZU�B
a
Mul_GPU_DT_HALF_DT_HALF_kernel*�2� 8��@��H��bmodel_3/dropout_18/dropout/MulhuZU�B
a
Mul_GPU_DT_HALF_DT_HALF_kernel*�2� 8��@��H��bmodel_3/dropout_19/dropout/MulhuZU�B
�
�void tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)�*  28��@��H��b3gradient_tape/model_3/conv2d_90/BiasAdd/BiasAddGradhuZU�B
K
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��bmul_54huZU�B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��b.gradient_tape/conv2d_95/kernel/Regularizer/MulhuZU�B
�
�void cudnn::bn_bw_1C11_singleread_specialized<__half2, 512, 1, 2, 7>(float, float, float, float, cudnnTensorStruct, __half2 const*, cudnnTensorStruct, __half2 const*, cudnnTensorStruct, __half2*, float const*, float*, float*, float const*, float const*, float, cudnn::reduced_divisor, int, cudnn::reduced_divisor, cudnn::bnBwPersistentState*, int, float, float, float, int, float, cudnnStatus_t*, bool)8� � *�2�
8��@��H��bAgradient_tape/model_3/batch_normalization_36/FusedBatchNormGradV3huZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*�2R8��@��H��Xb;gradient_tape/model_3/conv2d_95/Conv2D/Conv2DBackpropFilterhuZU�B
k
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��b#conv2d_95/kernel/Regularizer/SquarehuZU�B
�
htensorflow::functor::ReluGradHalfKernelVector(Eigen::half const*, Eigen::half const*, Eigen::half*, int)*�2�8��@��H��b,gradient_tape/model_3/activation_39/ReluGradhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*�2{8��@��H��bmodel_3/concatenate_30/concathuZU�B
�
�void cutlass_cudnn::Kernel<cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4> >(cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4>::Params)6* 2@8��@��H��Xb;gradient_tape/model_3/conv2d_88/Conv2D/Conv2DBackpropFilterhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8��@��H��b0gradient_tape/model_3/conv2d_95/Conv2D/Cast/CasthuZU�B
�
�void pooling_fw_kernel_max_nhwc<__half, float, 0, (cudnnNanPropagation_t)0, false>(PoolingParams, __half*, __half const*, float, float, int)*�2��8��@��H��b model_3/max_pooling2d_14/MaxPoolhu  �B
T
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�2�Z8��@��H��bIsFinite_50hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8��@��H��bmodel_3/conv2d_95/Conv2D/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�2{8��@��H��b,gradient_tape/model_3/concatenate_28/Slice_1huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�2{8��@��H��b,gradient_tape/model_3/concatenate_28/Slice_2huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�2{8��@��H��b,gradient_tape/model_3/concatenate_29/Slice_2huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�2{8��@��H��b*gradient_tape/model_3/concatenate_31/SlicehuZU�B
�
�void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>)*�2{8��@��H��b7model_3/dropout_19/dropout/random_uniform/RandomUniformhuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�2{8��@��H��b,gradient_tape/model_3/concatenate_31/Slice_1huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�2{8��@��H��b,gradient_tape/model_3/concatenate_29/Slice_1huZU�B
�
�void pooling_fw_kernel_max_nhwc<__half, float, 0, (cudnnNanPropagation_t)0, false>(PoolingParams, __half*, __half const*, float, float, int)*�2��8��@��H��b model_3/max_pooling2d_15/MaxPoolhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8��@��H��bAdam/gradients/AddN_32huZU�B
s
'GreaterEqual_GPU_DT_HALF_DT_BOOL_kernel*�2� 8��@��H��b'model_3/dropout_19/dropout/GreaterEqualhuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8��@��H��bAdam/gradients/AddN_27huZU�B
�
�void cutlass_cudnn::Kernel<cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4> >(cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4>::Params)6* 2@8��@��H��Xb;gradient_tape/model_3/conv2d_85/Conv2D/Conv2DBackpropFilterhu  �B
t
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��b/gradient_tape/dense_10/kernel/Regularizer/Mul_1huZU�B
�
�void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>)*�2{8��@��H��b7model_3/dropout_18/dropout/random_uniform/RandomUniformhuZU�B
s
'GreaterEqual_GPU_DT_HALF_DT_BOOL_kernel*�2� 8��@��H��b'model_3/dropout_18/dropout/GreaterEqualhuZU�B
�
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*�2R8��@��H��bmodel_3/conv2d_88/BiasAddhuZU�B
�
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*�2R8��@��H��bmodel_3/conv2d_93/BiasAddhuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8��@��H��bAdam/gradients/AddN_24huZU�B
�
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*�2R8��@��H��bmodel_3/conv2d_94/BiasAddhuZU�B
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��b0gradient_tape/conv2d_89/kernel/Regularizer/Mul_1huZU�B
�
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*�2R8��@��H��bmodel_3/conv2d_85/BiasAddhuZU�B
�
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*�2R8��@��H��bmodel_3/conv2d_89/BiasAddhuZU�B
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��b0gradient_tape/conv2d_92/kernel/Regularizer/Mul_1huZU�B
�
�void cudnn::bn_fw_tr_1C11_singleread_specialized<__half2, 512, 1, 2, 0>(cudnnTensorStruct, __half2 const*, cudnnTensorStruct, __half2*, float const*, float const*, float, float, float*, float*, float*, float*, float, float, cudnn::reduced_divisor, int, cudnn::reduced_divisor, cudnn::bnFwPersistentState*, int, float, float, float, int, float, float, cudnnStatus_t*, bool)(� ��*�2�
8��@��H��b/model_3/batch_normalization_36/FusedBatchNormV3hu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�2R8��@��H��b<cond_1/then/_10/cond_1/Adam/Adam/update_20/ResourceApplyAdamhuZU�B
�
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*�2R8��@��H��bmodel_3/conv2d_86/BiasAddhuZU�B
c
 ampere_fp16_sgemm_fp16_32x128_nn9��*�2� 8��@��H��Xbmodel_3/conv2d_75/Conv2DhuZU�B
�
�void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*�2�8��@��H��b conv2d_94/kernel/Regularizer/Sumhu  �B
�
�void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*�2R8��@��H��Xbmodel_3/conv2d_92/Conv2DhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�2R8��@��H��b<cond_1/then/_10/cond_1/Adam/Adam/update_34/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*�2R8��@��H��Xb:gradient_tape/model_3/conv2d_89/Conv2D/Conv2DBackpropInputhuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*�2R8��@��H��Xb:gradient_tape/model_3/conv2d_92/Conv2D/Conv2DBackpropInputhuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*�2R8��@��H��Xbmodel_3/conv2d_89/Conv2DhuZU�B
�	
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*�2{8��@��H��bmodel_3/activation_39/ReluhuZU�B
�
�void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nt_align1>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nt_align1::Params)� ��*�2�8��@��H��Xb;gradient_tape/model_3/conv2d_75/Conv2D/Conv2DBackpropFilterhugU�A
�

�
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8��@��H��bAdam/gradients/AddN_5huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8��@��H��bAdam/gradients/AddN_21huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *�2{8��@��H��b/gradient_tape/conv2d_89/kernel/Regularizer/TilehuZU�B
T
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�2�>8��@��H��bIsFinite_52hu  �B
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��b0gradient_tape/conv2d_86/kernel/Regularizer/Mul_1huZU�B
�
Nvoid cudnn::ops::scalePackedTensor_kernel<__half, float>(long, __half*, float)*�2��8��@��H��b:gradient_tape/model_3/max_pooling2d_14/MaxPool/MaxPoolGradhu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�2R8��@��H��b<cond_1/then/_10/cond_1/Adam/Adam/update_48/ResourceApplyAdamhuZU�B
�
Nvoid cudnn::ops::scalePackedTensor_kernel<__half, float>(long, __half*, float)*�2��8��@��H��b:gradient_tape/model_3/max_pooling2d_15/MaxPool/MaxPoolGradhu  �B
�
�void tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)�*  2 8��@��H��b3gradient_tape/model_3/conv2d_85/BiasAdd/BiasAddGradhuZU�B
�
�void tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)�*  2 8��@��H��b3gradient_tape/model_3/conv2d_86/BiasAdd/BiasAddGradhuZU�B
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��b0gradient_tape/conv2d_91/kernel/Regularizer/Mul_1huZU�B
�
�void tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)�*  2 8��@��H��b3gradient_tape/model_3/conv2d_89/BiasAdd/BiasAddGradhuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*�2R8��@��H��Xb:gradient_tape/model_3/conv2d_86/Conv2D/Conv2DBackpropInputhuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*�2R8��@��H��Xbmodel_3/conv2d_86/Conv2DhuZU�B
�
�void tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)�*  2 8��@��H��b3gradient_tape/model_3/conv2d_88/BiasAdd/BiasAddGradhuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8��@��H��bAdam/gradients/AddN_26huZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�2R8��@��H��b<cond_1/then/_10/cond_1/Adam/Adam/update_18/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�2R8��@��H��b<cond_1/then/_10/cond_1/Adam/Adam/update_26/ResourceApplyAdamhuZU�B
�
�void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*�2�8��@��H��b conv2d_95/kernel/Regularizer/Sumhu  �B
r
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��b-gradient_tape/dense_10/kernel/Regularizer/MulhuZU�B
K
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��bmul_46huZU�B
K
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��bmul_62huZU�B
j
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��b"dense_10/kernel/Regularizer/SquarehuZU�B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��b.gradient_tape/conv2d_92/kernel/Regularizer/MulhuZU�B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��b.gradient_tape/conv2d_89/kernel/Regularizer/MulhuZU�B
K
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��bmul_38huZU�B
c
Mul_GPU_DT_HALF_DT_HALF_kernel*�2�
8��@��H��b model_3/dropout_20/dropout/Mul_1huZU�B
k
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��b#conv2d_92/kernel/Regularizer/SquarehuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *�2{8��@��H��b/gradient_tape/conv2d_86/kernel/Regularizer/TilehuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*�2R8��@��H��Xbmodel_3/conv2d_91/Conv2DhuZU�B
q
Mul_GPU_DT_HALF_DT_HALF_kernel*�2�
8��@��H��b.gradient_tape/model_3/dropout_20/dropout/Mul_1huZU�B
k
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��b#conv2d_89/kernel/Regularizer/SquarehuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*�2R8��@��H��Xb;gradient_tape/model_3/conv2d_92/Conv2D/Conv2DBackpropFilterhuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*�2R8��@��H��Xb:gradient_tape/model_3/conv2d_91/Conv2D/Conv2DBackpropInputhuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*�2R8��@��H��Xb;gradient_tape/model_3/conv2d_89/Conv2D/Conv2DBackpropFilterhuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8��@��H��b/gradient_tape/model_3/dense_10/MatMul/Cast/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *�2{8��@��H��b/gradient_tape/conv2d_91/kernel/Regularizer/TilehuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8��@��H��b0gradient_tape/model_3/conv2d_89/Conv2D/Cast/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8��@��H��b0gradient_tape/model_3/conv2d_92/Conv2D/Cast/CasthuZU�B
q
Mul_GPU_DT_HALF_DT_HALF_kernel*�2�	8��@��H��b.gradient_tape/model_3/dropout_21/dropout/Mul_1huZU�B
c
Mul_GPU_DT_HALF_DT_HALF_kernel*�2�	8��@��H��b model_3/dropout_21/dropout/Mul_1huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8��@��H��bmodel_3/conv2d_89/Conv2D/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8��@��H��bmodel_3/conv2d_92/Conv2D/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�2{8��@��H��b*gradient_tape/model_3/concatenate_29/SlicehuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8��@��H��bmodel_3/dense_10/MatMul/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�2{8��@��H��b*gradient_tape/model_3/concatenate_28/SlicehuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�2{8��@��H��b*gradient_tape/model_3/concatenate_30/SlicehuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�2{8��@��H��b,gradient_tape/model_3/concatenate_30/Slice_1huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8��@��H��bmodel_3/dropout_20/dropout/CasthuZU�B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��b.gradient_tape/conv2d_86/kernel/Regularizer/MulhuZU�B
K
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��bmul_30huZU�B
�
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*�2R8��@��H��bmodel_3/conv2d_87/BiasAddhuZU�B
k
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��b#conv2d_86/kernel/Regularizer/SquarehuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*�2R8��@��H��Xb;gradient_tape/model_3/conv2d_86/Conv2D/Conv2DBackpropFilterhuZU�B
�
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*�2R8��@��H��bmodel_3/conv2d_91/BiasAddhuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8��@��H��bmodel_3/dropout_21/dropout/CasthuZU�B
�
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*�2R8��@��H��bmodel_3/conv2d_84/BiasAddhuZU�B
�
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*�2R8��@��H��bmodel_3/conv2d_90/BiasAddhuZU�B
K
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��bmul_44huZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*�2R8��@��H��Xb;gradient_tape/model_3/conv2d_91/Conv2D/Conv2DBackpropFilterhuZU�B
k
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��b#conv2d_91/kernel/Regularizer/SquarehuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8��@��H��b0gradient_tape/model_3/conv2d_86/Conv2D/Cast/CasthuZU�B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��b.gradient_tape/conv2d_91/kernel/Regularizer/MulhuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8��@��H��bmodel_3/conv2d_86/Conv2D/CasthuZU�B
T
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�2� 8��@��H��bIsFinite_60hu  �B
o
Mul_GPU_DT_HALF_DT_HALF_kernel*�2�
8��@��H��b,gradient_tape/model_3/dropout_20/dropout/MulhuZU�B
a
Mul_GPU_DT_HALF_DT_HALF_kernel*�2�
8��@��H��bmodel_3/dropout_20/dropout/MulhuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8��@��H��bmodel_3/conv2d_91/Conv2D/CasthuZU�B
T
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�2�8��@��H��bIsFinite_36hu  �B
T
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�2�8��@��H��bIsFinite_44hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8��@��H��bAdam/gradients/AddN_18huZU�B
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��b0gradient_tape/conv2d_83/kernel/Regularizer/Mul_1huZU�B
o
Mul_GPU_DT_HALF_DT_HALF_kernel*�2�	8��@��H��b,gradient_tape/model_3/dropout_21/dropout/MulhuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8��@��H��b0gradient_tape/model_3/conv2d_91/Conv2D/Cast/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�2{8��@��H��b.gradient_tape/dense_10/kernel/Regularizer/TilehuZU�B
�
�void tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)�*  2 8��@��H��b3gradient_tape/model_3/conv2d_84/BiasAdd/BiasAddGradhuZU�B
�
�void tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)�*  2 8��@��H��b3gradient_tape/model_3/conv2d_87/BiasAdd/BiasAddGradhuZU�B
a
Mul_GPU_DT_HALF_DT_HALF_kernel*�2�	8��@��H��bmodel_3/dropout_21/dropout/MulhuZU�B
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��b0gradient_tape/conv2d_88/kernel/Regularizer/Mul_1huZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*�2R8��@��H��Xb:gradient_tape/model_3/conv2d_83/Conv2D/Conv2DBackpropInputhuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8��@��H��bAdam/gradients/AddN_23huZU�B
�
Campere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x5_nn��� ��*�2 8��@��H��Xbmodel_3/dense_10/MatMulh
�
Campere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn��� ��*�2 8��@��H��Xb%gradient_tape/model_3/dense_10/MatMulh
�
�void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*�2�8��@��H��b conv2d_89/kernel/Regularizer/Sumhu  �B
�
�void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>)*�2{8��@��H��b7model_3/dropout_20/dropout/random_uniform/RandomUniformhuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*�2R8��@��H��Xbmodel_3/conv2d_83/Conv2DhuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*�2R8��@��H��Xbmodel_3/conv2d_88/Conv2DhuZU�B
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��b0gradient_tape/conv2d_93/kernel/Regularizer/Mul_1huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8��@��H��bAdam/gradients/AddN_28huZU�B
�
�void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*�2�8��@��H��b conv2d_92/kernel/Regularizer/Sumhu  �B
�
�void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*�2�8��@��H��bdense_10/kernel/Regularizer/Sumhu  �B
�
�void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*�2R8��@��H��Xb:gradient_tape/model_3/conv2d_88/Conv2D/Conv2DBackpropInputhuZU�B
s
'GreaterEqual_GPU_DT_HALF_DT_BOOL_kernel*�2�
8��@��H��b'model_3/dropout_20/dropout/GreaterEqualhuZU�B
T
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�2�8��@��H��bIsFinite_28hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *�2{8��@��H��b/gradient_tape/conv2d_83/kernel/Regularizer/TilehuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *�2{8��@��H��b/gradient_tape/conv2d_88/kernel/Regularizer/TilehuZU�B
�
�void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>)*�2{8��@��H��b7model_3/dropout_21/dropout/random_uniform/RandomUniformhuZU�B
�
�void nhwcAddPaddingKernel<__half, __half, float, true, (cudnnKernelDataType_t)0>(int, int, int, int, int, int, int, int, __half const*, __half*, int, int, int, int, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor)*�2�8��@�H��Xb;gradient_tape/model_3/conv2d_77/Conv2D/Conv2DBackpropFilterhu  �B
T
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�2�8��@��H��bIsFinite_42hu  �B
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��b0gradient_tape/conv2d_85/kernel/Regularizer/Mul_1huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8��@��H��bAdam/gradients/AddN_17huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8��@��H��bAdam/gradients/AddN_20huZU�B
�
�void cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *�2�8��@��H��bAll_50hu  �B
�
�void nhwcAddPaddingKernel<__half, __half, float, true, (cudnnKernelDataType_t)0>(int, int, int, int, int, int, int, int, __half const*, __half*, int, int, int, int, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor)*�2�8��@�H��Xb;gradient_tape/model_3/conv2d_76/Conv2D/Conv2DBackpropFilterhu  �B
s
'GreaterEqual_GPU_DT_HALF_DT_BOOL_kernel*�2�	8��@��H��b'model_3/dropout_21/dropout/GreaterEqualhuZU�B
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��b0gradient_tape/conv2d_82/kernel/Regularizer/Mul_1huZU�B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��b.gradient_tape/conv2d_83/kernel/Regularizer/MulhuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�2{8��@��H��b,gradient_tape/model_3/concatenate_30/Slice_2huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�2{8��@��H��b,gradient_tape/model_3/concatenate_31/Slice_2huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *�2{8��@��H��b/gradient_tape/conv2d_82/kernel/Regularizer/TilehuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *�2{8��@��H��b/gradient_tape/conv2d_93/kernel/Regularizer/TilehuZU�B
�
�void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*�2�8��@��H��b conv2d_91/kernel/Regularizer/Sumhu  �B
�
�void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*�2R8��@��H��Xb:gradient_tape/model_3/conv2d_85/Conv2D/Conv2DBackpropInputhuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*�2R8��@��H��Xbmodel_3/conv2d_82/Conv2DhuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*�2R8��@��H��Xbmodel_3/conv2d_85/Conv2DhuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*�2R8��@��H��Xb;gradient_tape/model_3/conv2d_83/Conv2D/Conv2DBackpropFilterhuZU�B
K
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��bmul_22huZU�B
k
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��b#conv2d_83/kernel/Regularizer/SquarehuZU�B
�
�void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*�2�8��@��H��b conv2d_86/kernel/Regularizer/Sumhu  �B
�
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*�2R8��@��H��bmodel_3/conv2d_92/BiasAddhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�2R8��@��H��b<cond_1/then/_10/cond_1/Adam/Adam/update_12/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*�2R8��@��H��Xb:gradient_tape/model_3/conv2d_82/Conv2D/Conv2DBackpropInputhuZU�B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��b.gradient_tape/conv2d_88/kernel/Regularizer/MulhuZU�B
K
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��bmul_36huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8��@��H��bmodel_3/conv2d_75/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8��@��H��bmodel_3/conv2d_83/Conv2D/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8��@��H��b0gradient_tape/model_3/conv2d_83/Conv2D/Cast/CasthuZU�B
�
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*�2R8��@��H��bmodel_3/conv2d_95/BiasAddhuZU�B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8�x@�xH�xb.gradient_tape/conv2d_93/kernel/Regularizer/MulhuZU�B
H
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8�x@�xH�xbmul_50huZU�B
h
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8�x@�xH�xb#conv2d_88/kernel/Regularizer/SquarehuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8�x@�xH�xbmodel_3/conv2d_88/Conv2D/CasthuZU�B
�
�void tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)�*  2 8�x@�xH�xb3gradient_tape/model_3/conv2d_92/BiasAdd/BiasAddGradhuZU�B
�
�void tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)�*  2 8�x@�xH�xb3gradient_tape/model_3/conv2d_95/BiasAdd/BiasAddGradhuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*�2R8�x@�xH�xXb;gradient_tape/model_3/conv2d_88/Conv2D/Conv2DBackpropFilterhuZU�B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8�p@�pH�pb.gradient_tape/conv2d_82/kernel/Regularizer/MulhuZU�B
h
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8�p@�pH�pb#conv2d_93/kernel/Regularizer/SquarehuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<unsigned char const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<unsigned char const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8�p@�pH�pbmodel_3/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorForcedEvalOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorForcedEvalOp<Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_log_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorForcedEvalOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorForcedEvalOp<Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_log_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long)2*�28�p@�pH�pb:categorical_crossentropy/softmax_cross_entropy_with_logitshuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�2R8�p@�pH�pb<cond_1/then/_10/cond_1/Adam/Adam/update_40/ResourceApplyAdamhuZU�B
�
�void xmma_cudnn::gemm::split_k_kernel<xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, xmma_cudnn::Col, 256, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 3> >(xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, xmma_cudnn::Col, 256, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 3>::Params)? ��*�2	8�p@�pH�pPXb;gradient_tape/model_3/conv2d_79/Conv2D/Conv2DBackpropFilterhuMUB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8�h@�hH�hb.gradient_tape/conv2d_85/kernel/Regularizer/MulhuZU�B
H
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8�h@�hH�hbmul_20huZU�B
H
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8�h@�hH�hbmul_28huZU�B
�
:ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nt���*�28�h@�hH�hb'gradient_tape/model_3/dense_10/MatMul_1hugU�A
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8�h@�hH�hbmodel_3/conv2d_93/Conv2D/CasthuZU�B
�
�void cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *�2�8�h@�hH�hbAll_52hu  �B
�
�void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*�2R8�h@�hH�hXb;gradient_tape/model_3/conv2d_82/Conv2D/Conv2DBackpropFilterhuZU�B
h
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8�g@�gH�gb#conv2d_82/kernel/Regularizer/SquarehuZU�B
Q
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�2�8�`@�`H�`bIsFinite_34hu  �B
Q
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�2�8�`@�`H�`bIsFinite_20hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8�`@�`H�`bmodel_3/conv2d_82/Conv2D/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8�`@�`H�`bmodel_3/conv2d_85/Conv2D/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8�`@�`H�`b0gradient_tape/model_3/conv2d_88/Conv2D/Cast/CasthuZU�B
h
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8�_@�_H�_b#conv2d_85/kernel/Regularizer/SquarehuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*�2R8�_@�_H�_Xb;gradient_tape/model_3/conv2d_85/Conv2D/Conv2DBackpropFilterhuZU�B
Q
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�2�
8�X@�XH�XbIsFinite_48hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8�X@�XH�Xb0gradient_tape/model_3/conv2d_93/Conv2D/Cast/CasthuZU�B
�
�void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*�2�8�X@�XH�Xb conv2d_83/kernel/Regularizer/Sumhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8�P@�PH�Pb0gradient_tape/model_3/conv2d_85/Conv2D/Cast/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *�2{8�P@�PH�Pb/gradient_tape/conv2d_80/kernel/Regularizer/TilehuZU�B
�
�void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*�2�8�P@�PH�Pb conv2d_88/kernel/Regularizer/Sumhu  �B
Q
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�2�	8�H@�HH�HbIsFinite_18hu  �B
Q
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�2�	8�H@�HH�HbIsFinite_26hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8�H@�HH�Hb0gradient_tape/model_3/conv2d_82/Conv2D/Cast/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8�H@�HH�HbAdam/gradients/AddN_15huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorReshapingOp<Eigen::IndexList<int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> >, Eigen::TensorReductionOp<Eigen::internal::MaxReducer<float, 0>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorReshapingOp<Eigen::IndexList<int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> >, Eigen::TensorReductionOp<Eigen::internal::MaxReducer<float, 0>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long)@*�28�H@�HH�Hb:categorical_crossentropy/softmax_cross_entropy_with_logitshuZU�B
�
�void cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *�2�8�H@�HH�HbAll_44hu  �B
�
�void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*�2�8�H@�HH�Hb conv2d_82/kernel/Regularizer/Sumhu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�2R8�H@�HH�Hb<cond_1/then/_10/cond_1/Adam/Adam/update_10/ResourceApplyAdamhuZU�B
q
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2d8�@@�@H�@b0gradient_tape/conv2d_80/kernel/Regularizer/Mul_1huZU�B
�
�void cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *�2�8�@@�@H�@bAll_28hu  �B
�
�void cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *�2�8�@@�@H�@bAll_36hu  �B
�
�void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*�2�8�@@�@H�@b conv2d_85/kernel/Regularizer/Sumhu  �B
�
�void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*�2�8�@@�@H�@b conv2d_93/kernel/Regularizer/Sumhu  �B
�
�void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*�2R8�@@�@H�@Xb:gradient_tape/model_3/conv2d_80/Conv2D/Conv2DBackpropInputhuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*�2R8�@@�@H�@Xbmodel_3/conv2d_80/Conv2DhuZU�B
q
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2P8�8@�8H�8b0gradient_tape/conv2d_90/kernel/Regularizer/Mul_1huZU�B
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2d8�8@�8H�8b.gradient_tape/conv2d_80/kernel/Regularizer/MulhuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8�8@�8H�8bAdam/gradients/AddN_25huZU�B
�
�void cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *�2�8�8@�8H�8bAll_42hu  �B
�
�void cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *�2�8�8@�8H�8bAll_60hu  �B
�
�void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*�2R8�8@�8H�8Xb:gradient_tape/model_3/conv2d_79/Conv2D/Conv2DBackpropInputhuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*�2R8�8@�8H�8Xbmodel_3/conv2d_79/Conv2DhuZU�B
q
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2H8�0@�0H�0b0gradient_tape/conv2d_79/kernel/Regularizer/Mul_1huZU�B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2d8�0@�0H�0bmul_14huZU�B
g
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*�2d8�0@�0H�0b#conv2d_80/kernel/Regularizer/SquarehuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8�0@�0H�0bAdam/gradients/AddN_14huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *�28�0@�0H�0b/gradient_tape/conv2d_78/kernel/Regularizer/TilehuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *�2{8�0@�0H�0b/gradient_tape/conv2d_79/kernel/Regularizer/TilehuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *�2{8�0@�0H�0b/gradient_tape/conv2d_90/kernel/Regularizer/TilehuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorReshapingOp<Eigen::IndexList<int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> >, Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorReshapingOp<Eigen::IndexList<int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> >, Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long)/*�28�0@�0H�0b:categorical_crossentropy/softmax_cross_entropy_with_logitshuZU�B
�
�void cutlass_cudnn::Kernel<cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4> >(cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4>::Params)6* 28�0@�0H�0Xb;gradient_tape/model_3/conv2d_77/Conv2D/Conv2DBackpropFilterhu  �B
�
�void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*) *�28�0@�0H�0Xb;gradient_tape/model_3/conv2d_75/Conv2D/Conv2DBackpropFilterhu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�2P8�0@�0H�0b<cond_1/then/_10/cond_1/Adam/Adam/update_32/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2�8�0@�0H�0b3gradient_tape/model_3/conv2d_79/BiasAdd/BiasAddGradhuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*�2R8�0@�0H�0Xb;gradient_tape/model_3/conv2d_80/Conv2D/Conv2DBackpropFilterhuZU�B
Q
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�2�8�(@�(H�(bIsFinite_40hu  �B
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2H8�(@�(H�(b.gradient_tape/conv2d_79/kernel/Regularizer/MulhuZU�B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2H8�(@�(H�(bmul_12huZU�B
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2P8�(@�(H�(b.gradient_tape/conv2d_90/kernel/Regularizer/MulhuZU�B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2P8�(@�(H�(bmul_42huZU�B
g
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*�2H8�(@�(H�(b#conv2d_79/kernel/Regularizer/SquarehuZU�B
g
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*�2P8�(@�(H�(b#conv2d_90/kernel/Regularizer/SquarehuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8�(@�(H�(b0gradient_tape/model_3/conv2d_80/Conv2D/Cast/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long) *�28�(@�H�b(ArithmeticOptimizer/AddOpsRewrite_AddN_1huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *�28�(@�(H�(b/gradient_tape/conv2d_77/kernel/Regularizer/TilehuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *�2@8�(@�(H�(b/gradient_tape/conv2d_81/kernel/Regularizer/TilehuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *�2P8�(@�(H�(b/gradient_tape/conv2d_87/kernel/Regularizer/TilehuZU�B
�
�void cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *�2�8�(@�(H�(bAll_18hu  �B
�
�void cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *�2�8�(@�(H�(bAll_26hu  �B
�
�void cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *�2�8�(@�(H�(bAll_48hu  �B
�
�void cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *�2�8�(@�(H�(bAll_34hu  �B
�
�void cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *�2�8�(@�(H�(bAll_20hu  �B
�
�void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*�2P8�(@�(H�(b conv2d_90/kernel/Regularizer/Sumhu  �B
�
�void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*�2d8�(@�(H�(b conv2d_80/kernel/Regularizer/Sumhu  �B
�
�void cutlass::Kernel<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2>(cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2::Params)_ ��*�28�(@�(H�(Xbmodel_3/dense_11/MatMulhugU�A
�
�void cutlass_cudnn::Kernel<cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4> >(cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4>::Params)6* 28�(@�(H�(Xb;gradient_tape/model_3/conv2d_76/Conv2D/Conv2DBackpropFilterhu  �B
�
�void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*) *�2�8�(@�(H�(Xb;gradient_tape/model_3/conv2d_90/Conv2D/Conv2DBackpropFilterhu  �B
�
�void tensorflow::(anonymous namespace)::GenerateNormalizedProb<Eigen::half, float, 8>(Eigen::half const*, float const*, Eigen::half const*, Eigen::half*, int, int, bool)*�28�(@�(H�(bmodel_3/activation_43/Softmaxhu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�(@�(H�(b<cond_1/then/_10/cond_1/Adam/Adam/update_37/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�2@8�(@�(H�(b<cond_1/then/_10/cond_1/Adam/Adam/update_16/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�2@8�(@�(H�(b<cond_1/then/_10/cond_1/Adam/Adam/update_24/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *�28�(@�(H�(bAll_57hu  �B
�
�void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *�28�(@�(H�(bAll_61hu  �B
�
�void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *�2 8�(@�(H�(bAll_16hu  �B
�
�void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *�2 8�(@�(H�(bAll_24hu  �B
�
�void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2�8�(@�(H�(b3gradient_tape/model_3/conv2d_82/BiasAdd/BiasAddGradhuZU�B
�
�void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2�8�(@�(H�(b3gradient_tape/model_3/conv2d_85/BiasAdd/BiasAddGradhuZU�B
�
�void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2�8�(@�(H�(b3gradient_tape/model_3/conv2d_86/BiasAdd/BiasAddGradhuZU�B
�
�void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2�8�(@�(H�(b3gradient_tape/model_3/conv2d_88/BiasAdd/BiasAddGradhuZU�B
�
�void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2�8�(@�(H�(b3gradient_tape/model_3/conv2d_89/BiasAdd/BiasAddGradhuZU�B
�
�void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2�8�(@�(H�(b3gradient_tape/model_3/conv2d_92/BiasAdd/BiasAddGradhuZU�B
�
�void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2�8�(@�(H�(b3gradient_tape/model_3/conv2d_95/BiasAdd/BiasAddGradhuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*�2R8�(@�(H�(Xb;gradient_tape/model_3/conv2d_79/Conv2D/Conv2DBackpropFilterhuZU�B
�
�void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *�2 8�'@�'H�'bAll_32hu  �B
Q
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�2�8� @� H� bIsFinite_10hu  �B
Q
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�2�8� @� H� bIsFinite_12hu  �B
q
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28� @� H� b0gradient_tape/conv2d_87/kernel/Regularizer/Mul_1huZU�B
l
 ampere_fp16_sgemm_fp16_128x32_tn9��*�28� @� H� Xb%gradient_tape/model_3/dense_11/MatMulhuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8� @� H� b0gradient_tape/model_3/conv2d_79/Conv2D/Cast/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8� @� H� b0gradient_tape/model_3/conv2d_90/Conv2D/Cast/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2@8� @� H� bAdam/gradients/AddN_16huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2@8� @� H� bAdam/gradients/AddN_19huZU�B
�	
�	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2@8� @� H� bAdam/gradients/AddN_1huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�28� @� H� b.gradient_tape/dense_11/kernel/Regularizer/TilehuZU�B
�	
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_squared_difference_op<float>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_squared_difference_op<float>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, int) *�2@8� @� H� b8model_3/batch_normalization_38/moments/SquaredDifferencehuZU�B
�	
�	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)&*�28� @� H� b:categorical_crossentropy/softmax_cross_entropy_with_logitshuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_quotient_op<float, float>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_quotient_op<float, float>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long).*�28� @� H� b:categorical_crossentropy/softmax_cross_entropy_with_logitshuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *�28� @� H� b/gradient_tape/conv2d_75/kernel/Regularizer/TilehuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorTupleReducerOp<Eigen::internal::ArgMaxTupleReducer<Eigen::Tuple<long, float> >, Eigen::array<long, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorTupleReducerOp<Eigen::internal::ArgMaxTupleReducer<Eigen::Tuple<long, float> >, Eigen::array<long, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long) *�28� @� H� bArgMaxhuZU�B
�
�void cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *�2$8� @� H� bAll_10hu  �B
�
�void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*�2H8� @� H� b conv2d_79/kernel/Regularizer/Sumhu  �B
�
�void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *�28� @� H� bAll_52hu���B
�
�void cutlass::Kernel<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nt_align2>(cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nt_align2::Params)_ ��*�2@8� @� H� b'gradient_tape/model_3/dense_11/MatMul_1hugU�A
�
�void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*) *�2�8� @� H� Xb;gradient_tape/model_3/conv2d_81/Conv2D/Conv2DBackpropFilterhu  �B
�
�void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*) *�2�8� @� H� Xb;gradient_tape/model_3/conv2d_87/Conv2D/Conv2DBackpropFilterhu  �B
�
�void splitKreduce_kernel<float, __half, float, __half>(cublasSplitKParams<float>, float const*, __half const*, __half*, float const*, float const*, __half const*) *�2�8� @� H� Xb;gradient_tape/model_3/conv2d_78/Conv2D/Conv2DBackpropFilterhu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28� @� H� b9cond_1/then/_10/cond_1/Adam/Adam/update/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28� @� H� b;cond_1/then/_10/cond_1/Adam/Adam/update_1/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28� @� H� b<cond_1/then/_10/cond_1/Adam/Adam/update_13/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28� @� H� b<cond_1/then/_10/cond_1/Adam/Adam/update_19/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28� @� H� b<cond_1/then/_10/cond_1/Adam/Adam/update_21/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28� @� H� b<cond_1/then/_10/cond_1/Adam/Adam/update_22/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28� @� H� b<cond_1/then/_10/cond_1/Adam/Adam/update_27/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28� @� H� b<cond_1/then/_10/cond_1/Adam/Adam/update_29/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28� @� H� b<cond_1/then/_10/cond_1/Adam/Adam/update_31/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28� @� H� b<cond_1/then/_10/cond_1/Adam/Adam/update_33/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28� @� H� b<cond_1/then/_10/cond_1/Adam/Adam/update_35/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28� @� H� b<cond_1/then/_10/cond_1/Adam/Adam/update_38/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28� @� H� b<cond_1/then/_10/cond_1/Adam/Adam/update_41/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28� @� H� b<cond_1/then/_10/cond_1/Adam/Adam/update_45/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28� @� H� b<cond_1/then/_10/cond_1/Adam/Adam/update_49/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28� @� H� b;cond_1/then/_10/cond_1/Adam/Adam/update_5/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28� @� H� b<cond_1/then/_10/cond_1/Adam/Adam/update_51/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28� @� H� b<cond_1/then/_10/cond_1/Adam/Adam/update_53/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28� @� H� b;cond_1/then/_10/cond_1/Adam/Adam/update_6/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28� @� H� b;cond_1/then/_10/cond_1/Adam/Adam/update_9/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28� @� H� b<cond_1/then/_10/cond_1/Adam/Adam/update_46/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28� @� H� b<cond_1/then/_10/cond_1/Adam/Adam/update_47/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28� @� H� b<cond_1/then/_10/cond_1/Adam/Adam/update_58/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28� @� H� b<cond_1/then/_10/cond_1/Adam/Adam/update_61/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28� @� H� b<cond_1/then/_10/cond_1/Adam/Adam/update_63/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28� @� H� b<cond_1/then/_10/cond_1/Adam/Adam/update_54/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28� @� H� b<cond_1/then/_10/cond_1/Adam/Adam/update_55/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28� @� H� b;cond_1/then/_10/cond_1/Adam/Adam/update_4/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28� @� H� b<cond_1/then/_10/cond_1/Adam/Adam/update_64/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28� @� H� b;cond_1/then/_10/cond_1/Adam/Adam/update_8/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *�28� @� H� bAll_38hu  �B
�
�void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *�28� @� H� bAll_46hu  �B
�
�void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *�28� @� H� bAll_47hu  �B
�
�void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *�28� @� H� bAll_51hu  �B
�
�void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *�28� @� H� bAll_54hu  �B
�
�void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *�28� @� H� bAll_55hu  �B
�
�void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *�28� @� H� bAll_63hu  �B
�
�void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*�2 8� @� H� b conv2d_87/kernel/Regularizer/Sumhu  �B
�
�void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2�8� @� H� b3gradient_tape/model_3/conv2d_78/BiasAdd/BiasAddGradhuZU�B
�
�void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2�8� @� H� b3gradient_tape/model_3/conv2d_80/BiasAdd/BiasAddGradhuZU�B
�
�void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2�8� @� H� b3gradient_tape/model_3/conv2d_81/BiasAdd/BiasAddGradhuZU�B
�
�void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2�8� @� H� b3gradient_tape/model_3/conv2d_83/BiasAdd/BiasAddGradhuZU�B
�
�void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2�8� @� H� b3gradient_tape/model_3/conv2d_87/BiasAdd/BiasAddGradhuZU�B
�
�void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>)*�2@8� @� H� b7model_3/dropout_22/dropout/random_uniform/RandomUniformhuZU�B
�
�void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>)*�2@8� @� H� b7model_3/dropout_23/dropout/random_uniform/RandomUniformhuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*�28� @� H� Xbmodel_3/conv2d_77/Conv2DhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b<cond_1/then/_10/cond_1/Adam/Adam/update_23/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b<cond_1/then/_10/cond_1/Adam/Adam/update_43/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b<cond_1/then/_10/cond_1/Adam/Adam/update_57/ResourceApplyAdamhuZU�B
o
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b,model_3/batch_normalization_38/batchnorm/addhuZU�B
q
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�2@8�@�H�b.model_3/batch_normalization_38/batchnorm/add_1huZU�B
q
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�2@8�@�H�b.model_3/batch_normalization_39/batchnorm/add_1huZU�B
}
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*�2@8�@�H�b<gradient_tape/model_3/batch_normalization_38/moments/truedivhuZU�B

 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*�2@8�@�H�b>gradient_tape/model_3/batch_normalization_39/moments/truediv_1huZU�B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�28�@�H�b
IsFinite_1hu  �B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�28�@�H�bIsFinite_15hu  �B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�28�@�H�bIsFinite_19hu  �B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�28�@�H�bIsFinite_21hu  �B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�28�@�H�bIsFinite_31hu  �B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�28�@�H�bIsFinite_37hu  �B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�28�@�H�bIsFinite_38hu  �B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�28�@�H�b
IsFinite_9hu  �B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�28�@�H�bIsFinite_47hu  �B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�28�@�H�bIsFinite_59hu  �B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�28�@�H�bIsFinite_61hu  �B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�28�@�H�bIsFinite_55hu  �B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�2@8�@�H�bIsFinite_16hu  �B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�2@8�@�H�bIsFinite_24hu  �B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�2P8�@�H�bIsFinite_32hu  �B
D
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bMulhuZU�B
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b conv2d_76/kernel/Regularizer/mulhuZU�B
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b conv2d_77/kernel/Regularizer/mulhuZU�B
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b conv2d_83/kernel/Regularizer/mulhuZU�B
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b conv2d_85/kernel/Regularizer/mulhuZU�B
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b conv2d_86/kernel/Regularizer/mulhuZU�B
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b conv2d_92/kernel/Regularizer/mulhuZU�B
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bdense_11/kernel/Regularizer/mulhuZU�B
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b.gradient_tape/conv2d_76/kernel/Regularizer/MulhuZU�B
�
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b@gradient_tape/model_3/batch_normalization_38/batchnorm/mul/Mul_1huZU�B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b2model_3/batch_normalization_38/AssignMovingAvg/mulhuZU�B
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b4model_3/batch_normalization_39/AssignMovingAvg_1/mulhuZU�B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bmul_11huZU�B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bmul_15huZU�B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bmul_16huZU�B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bmul_21huZU�B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bmul_23huZU�B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bmul_24huZU�B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bmul_27huZU�B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bmul_29huZU�B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bmul_33huZU�B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bmul_35huZU�B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bmul_37huZU�B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bmul_39huZU�B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bmul_40huZU�B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bmul_43huZU�B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bmul_45huZU�B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bmul_47huZU�B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bmul_48huZU�B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bmul_49huZU�B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bmul_55huZU�B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bmul_56huZU�B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bmul_59huZU�B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bmul_63huZU�B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bmul_67huZU�B
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bmul_8huZU�B
q
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b0gradient_tape/conv2d_75/kernel/Regularizer/Mul_1huZU�B
�
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b@gradient_tape/model_3/batch_normalization_38/batchnorm/mul_2/MulhuZU�B
m
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b,model_3/batch_normalization_38/batchnorm/mulhuZU�B
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b.model_3/batch_normalization_38/batchnorm/mul_2huZU�B
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b.model_3/batch_normalization_39/batchnorm/mul_2huZU�B
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bmul_6huZU�B
q
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b0gradient_tape/conv2d_77/kernel/Regularizer/Mul_1huZU�B
n
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b-gradient_tape/dense_11/kernel/Regularizer/MulhuZU�B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b/gradient_tape/dense_11/kernel/Regularizer/Mul_1huZU�B
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b.gradient_tape/conv2d_78/kernel/Regularizer/MulhuZU�B
q
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b0gradient_tape/conv2d_78/kernel/Regularizer/Mul_1huZU�B
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b.gradient_tape/conv2d_81/kernel/Regularizer/MulhuZU�B
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b.gradient_tape/conv2d_84/kernel/Regularizer/MulhuZU�B
y
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b8gradient_tape/model_3/batch_normalization_38/moments/MulhuZU�B
y
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b8gradient_tape/model_3/batch_normalization_39/moments/MulhuZU�B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bmul_18huZU�B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bmul_26huZU�B
q
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b0gradient_tape/conv2d_81/kernel/Regularizer/Mul_1huZU�B
q
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b0gradient_tape/conv2d_84/kernel/Regularizer/Mul_1huZU�B
�
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bBgradient_tape/model_3/batch_normalization_38/batchnorm/mul_1/Mul_1huZU�B
{
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b:gradient_tape/model_3/batch_normalization_38/moments/mul_1huZU�B
�
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bBgradient_tape/model_3/batch_normalization_39/batchnorm/mul_1/Mul_1huZU�B
{
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b:gradient_tape/model_3/batch_normalization_39/moments/mul_1huZU�B
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b.gradient_tape/conv2d_87/kernel/Regularizer/MulhuZU�B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bmul_34huZU�B
�
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2@8�@�H�b@gradient_tape/model_3/batch_normalization_38/batchnorm/mul_1/MulhuZU�B
�
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2@8�@�H�b@gradient_tape/model_3/batch_normalization_39/batchnorm/mul_1/MulhuZU�B
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2@8�@�H�b.model_3/batch_normalization_38/batchnorm/mul_1huZU�B
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2@8�@�H�b.model_3/batch_normalization_39/batchnorm/mul_1huZU�B
k
Mul_GPU_DT_HALF_DT_HALF_kernel*�28�@�H�b,gradient_tape/model_3/dropout_22/dropout/MulhuZU�B
m
Mul_GPU_DT_HALF_DT_HALF_kernel*�28�@�H�b.gradient_tape/model_3/dropout_22/dropout/Mul_1huZU�B
k
Mul_GPU_DT_HALF_DT_HALF_kernel*�28�@�H�b,gradient_tape/model_3/dropout_23/dropout/MulhuZU�B
m
Mul_GPU_DT_HALF_DT_HALF_kernel*�28�@�H�b.gradient_tape/model_3/dropout_23/dropout/Mul_1huZU�B
]
Mul_GPU_DT_HALF_DT_HALF_kernel*�28�@�H�bmodel_3/dropout_22/dropout/MulhuZU�B
_
Mul_GPU_DT_HALF_DT_HALF_kernel*�28�@�H�b model_3/dropout_23/dropout/Mul_1huZU�B

 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b>gradient_tape/model_3/batch_normalization_38/batchnorm/sub/Neghu  �B

 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b>gradient_tape/model_3/batch_normalization_39/batchnorm/sub/Neghu  �B
`
 Pow_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bcond_1/then/_10/cond_1/Adam/PowhuZU�B
q
"Rsqrt_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b.model_3/batch_normalization_38/batchnorm/Rsqrthu  �B
g
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b#conv2d_75/kernel/Regularizer/SquarehuZU�B
g
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b#conv2d_81/kernel/Regularizer/SquarehuZU�B
g
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b#conv2d_84/kernel/Regularizer/SquarehuZU�B
g
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b#conv2d_87/kernel/Regularizer/SquarehuZU�B
s
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b2model_3/batch_normalization_38/AssignMovingAvg/subhuZU�B
s
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b2model_3/batch_normalization_39/AssignMovingAvg/subhuZU�B
y
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�2@8�@�H�b8gradient_tape/model_3/batch_normalization_38/moments/subhuZU�B
�
htensorflow::functor::ReluGradHalfKernelVector(Eigen::half const*, Eigen::half const*, Eigen::half*, int)*�28�@�H�b,gradient_tape/model_3/activation_41/ReluGradhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2@8�@�H�bmodel_3/dropout_22/dropout/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bCasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�b}categorical_crossentropy/softmax_cross_entropy_with_logits/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Cast_2huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�b�gradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/Cast/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bmodel_3/conv2d_76/BiasAdd/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bmodel_3/conv2d_77/BiasAdd/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bmodel_3/conv2d_78/BiasAdd/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bmodel_3/conv2d_84/BiasAdd/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bmodel_3/conv2d_88/BiasAdd/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bmodel_3/conv2d_91/BiasAdd/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bmodel_3/conv2d_93/BiasAdd/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bmodel_3/conv2d_95/BiasAdd/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bmodel_3/conv2d_78/Conv2D/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2@8�@�H�b6gradient_tape/model_3/batch_normalization_39/Cast/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2@8�@�H�b%model_3/batch_normalization_38/Cast_1huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2@8�@�H�b%model_3/batch_normalization_39/Cast_1huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2@8�@�H�bmodel_3/conv2d_81/Conv2D/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2@8�@�H�bmodel_3/conv2d_84/Conv2D/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8�@�H�bmodel_3/conv2d_79/Conv2D/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8�@�H�bmodel_3/conv2d_80/Conv2D/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2{8�@�H�bmodel_3/conv2d_90/Conv2D/CasthuZU�B
�	
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*�2@8�@�H�bmodel_3/activation_41/ReluhuZU�B
�	
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*�2@8�@�H�bmodel_3/activation_42/ReluhuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�28�@�H�b;gradient_tape/categorical_crossentropy/weighted_loss/Tile_1huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�b
div_no_nanhuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bEgradient_tape/categorical_crossentropy/weighted_loss/value/div_no_nanhuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_inverse_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_inverse_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�28�@�H�btruedivhuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�b1gradient_tape/model_3/conv2d_75/BiasAdd/Cast/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�b1gradient_tape/model_3/conv2d_78/BiasAdd/Cast/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�b1gradient_tape/model_3/conv2d_81/BiasAdd/Cast/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�b1gradient_tape/model_3/conv2d_82/BiasAdd/Cast/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�b1gradient_tape/model_3/conv2d_88/BiasAdd/Cast/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�b1gradient_tape/model_3/conv2d_91/BiasAdd/Cast/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�b1gradient_tape/model_3/conv2d_95/BiasAdd/Cast/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�b0gradient_tape/model_3/conv2d_77/Conv2D/Cast/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2@8�@�H�b8gradient_tape/model_3/batch_normalization_38/Cast_1/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2@8�@�H�b0gradient_tape/model_3/conv2d_84/Conv2D/Cast/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2@8�@�H�b#model_3/batch_normalization_39/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2P8�@�H�b0gradient_tape/model_3/conv2d_87/Conv2D/Cast/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bCast_7huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�b8categorical_crossentropy/weighted_loss/num_elements/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�b9gradient_tape/model_3/batch_normalization_38/moments/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�b"cond_1/then/_10/cond_1/Adam/Cast_1huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�b.model_3/batch_normalization_38/AssignMovingAvghuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bAdam/gradients/AddN_11huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bAdam/gradients/AddN_2huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bAdam/gradients/AddN_12huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bAdam/gradients/AddN_33huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bAdam/gradients/AddN_13huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2P8�@�H�bAdam/gradients/AddN_22huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bAssignAddVariableOphuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bAssignAddVariableOp_2huZU�B
�	
�	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2@8�@�H�bAdam/gradients/AddN_3huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long) *�28�@�H�b(ArithmeticOptimizer/AddOpsRewrite_AddN_1huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�2@8�@�H�b@gradient_tape/model_3/batch_normalization_38/moments/BroadcastTohuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�2@8�@�H�bBgradient_tape/model_3/batch_normalization_38/moments/BroadcastTo_1huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�2@8�@�H�b@gradient_tape/model_3/batch_normalization_39/moments/BroadcastTohuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�2@8�@�H�bBgradient_tape/model_3/batch_normalization_39/moments/BroadcastTo_1huZU�B
�	
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_squared_difference_op<float>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_squared_difference_op<float>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, int) *�2@8�@�H�b8model_3/batch_normalization_39/moments/SquaredDifferencehuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *�28�@�H�b/gradient_tape/conv2d_76/kernel/Regularizer/TilehuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bAssignAddVariableOp_4huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�b>cond/then/_0/cond/cond/else/_843/cond/cond/AssignAddVariableOphuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorTupleReducerOp<Eigen::internal::ArgMaxTupleReducer<Eigen::Tuple<long, float> >, Eigen::array<long, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorTupleReducerOp<Eigen::internal::ArgMaxTupleReducer<Eigen::Tuple<long, float> >, Eigen::array<long, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long) *�28�@�H�bArgMax_1huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorEvalToOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorForcedEvalOp<Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_log_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorEvalToOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorForcedEvalOp<Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_log_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>, long)(*�28�@�H�b:categorical_crossentropy/softmax_cross_entropy_with_logitshuZU�B
�
�void cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *�2(8�@�H�bAll_40hu  �B
�
�void cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *�228�@�H�bAll_12hu  �B
�
�void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *�28�@�H�bAll_18hu���B
�
�void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *�28�@�H�bAll_20hu���B
�
�void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *�28�@�H�bAll_28hu���B
�
�void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *�28�@�H�bAll_34hu���B
�
�void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *�28�@�H�bAll_36hu���B
�
�void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *�28�@�H�bAll_40hu���B
�
�void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *�28�@�H�bAll_44hu���B
�
�void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *�28�@�H�bAll_48hu���B
�
�void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *�28�@�H�bAll_50hu���B
�
�void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *�28�@�H�bAll_56hu���B
�
�void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *�28�@�H�bAll_60hu���B
�
�void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float>, float>(float*, float*, int, tensorflow::functor::Sum<float>, float)0*�28�@�H�b conv2d_80/kernel/Regularizer/Sumhu  �B
�
�void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float>, float>(float*, float*, int, tensorflow::functor::Sum<float>, float)0*�28�@�H�b conv2d_83/kernel/Regularizer/Sumhu  �B
�
�void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float>, float>(float*, float*, int, tensorflow::functor::Sum<float>, float)0*�28�@�H�b conv2d_86/kernel/Regularizer/Sumhu  �B
�
�void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float>, float>(float*, float*, int, tensorflow::functor::Sum<float>, float)0*�28�@�H�b conv2d_91/kernel/Regularizer/Sumhu  �B
�
�void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float>, float>(float*, float*, int, tensorflow::functor::Sum<float>, float)0*�28�@�H�b conv2d_94/kernel/Regularizer/Sumhu  �B
�
�void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float>, float>(float*, float*, int, tensorflow::functor::Sum<float>, float)0*�28�@�H�b conv2d_95/kernel/Regularizer/Sumhu  �B
�
�void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float>, float>(float*, float*, int, tensorflow::functor::Sum<float>, float)0*�28�@�H�bdense_10/kernel/Regularizer/Sumhu  �B
�
�void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float>, float>(float*, float*, int, tensorflow::functor::Sum<float>, float)0*�28�@�H�bdense_9/kernel/Regularizer/Sumhu  �B
�
�void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*) *�2�8�@�H�Xb;gradient_tape/model_3/conv2d_84/Conv2D/Conv2DBackpropFilterhu  �B
�
�void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*) *�2�8�@�H�Xbmodel_3/dense_9/MatMulhu  �B
�
�void tensorflow::(anonymous namespace)::concat_fixed_kernel<bool, int>(tensorflow::GpuDeviceArrayStruct<bool const*, 8>, int, int, int, bool*)*B28�@�H�bAll_66/inputhu��B
�
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*�2@8�@�H�bmodel_3/dense_10/BiasAddhuZU�B
�
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*�2@8�@�H�bmodel_3/dense_9/BiasAddhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b<cond_1/then/_10/cond_1/Adam/Adam/update_11/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b<cond_1/then/_10/cond_1/Adam/Adam/update_14/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b<cond_1/then/_10/cond_1/Adam/Adam/update_15/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b<cond_1/then/_10/cond_1/Adam/Adam/update_17/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b;cond_1/then/_10/cond_1/Adam/Adam/update_2/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b<cond_1/then/_10/cond_1/Adam/Adam/update_25/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b<cond_1/then/_10/cond_1/Adam/Adam/update_30/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b<cond_1/then/_10/cond_1/Adam/Adam/update_39/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b<cond_1/then/_10/cond_1/Adam/Adam/update_65/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b;cond_1/then/_10/cond_1/Adam/Adam/update_7/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b<cond_1/then/_10/cond_1/Adam/Adam/update_59/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b<cond_1/then/_10/cond_1/Adam/Adam/update_62/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *�28�@�H�bAllhu  �B
�
�void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *�28�@�H�bAll_1hu  �B
�
�void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *�28�@�H�bAll_13hu  �B
�
�void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *�28�@�H�bAll_14hu  �B
�
�void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *�28�@�H�bAll_15hu  �B
�
�void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *�28�@�H�bAll_2hu  �B
�
�void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *�28�@�H�bAll_22hu  �B
�
�void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *�28�@�H�bAll_23hu  �B
�
�void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *�28�@�H�bAll_3hu  �B
�
�void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *�28�@�H�bAll_30hu  �B
�
�void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *�28�@�H�bAll_31hu  �B
�
�void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *�28�@�H�bAll_39hu  �B
�
�void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *�28�@�H�bAll_41hu  �B
�
�void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *�28�@�H�bAll_53hu  �B
�
�void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *�28�@�H�bAll_58hu  �B
�
�void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *�28�@�H�bAll_59hu  �B
�
�void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *�28�@�H�bAll_62hu  �B
�
�void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *�28�@�H�bAll_65hu  �B
�
�void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *�28�@�H�bAll_66hu  �B
�
�void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *�28�@�H�bAll_9hu  �B
�
�void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *�2 8�@�H�bAll_64hu  �B
�
�void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *�2 8�@�H�bAll_8hu  �B
�
�void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*�28�@�H�b conv2d_75/kernel/Regularizer/Sumhu  �B
�
�void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*�28�@�H�b conv2d_76/kernel/Regularizer/Sumhu  �B
�
�void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*�28�@�H�b conv2d_77/kernel/Regularizer/Sumhu  �B
�
�void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*�2 8�@�H�b conv2d_81/kernel/Regularizer/Sumhu  �B
�
�void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*�2 8�@�H�b conv2d_84/kernel/Regularizer/Sumhu  �B
�
�void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*�2 8�@�H�bdense_11/kernel/Regularizer/Sumhu  �B
�
�void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2 8�@�H�b3gradient_tape/model_3/conv2d_75/BiasAdd/BiasAddGradhuZU�B
�
�void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2@8�@�H�b3gradient_tape/model_3/conv2d_77/BiasAdd/BiasAddGradhuZU�B
�
�void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2�8�@�H�b3gradient_tape/model_3/conv2d_84/BiasAdd/BiasAddGradhuZU�B
�
�void tensorflow::functor::CleanupSegments<bool*, bool*, tensorflow::functor::And>(bool*, bool*, int, int, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type)* 28�@�H�bAll_32huMUB
�
�void tensorflow::functor::CleanupSegments<bool*, bool*, tensorflow::functor::And>(bool*, bool*, int, int, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type)* 28�@�H�bAll_4huMUB
�
�void tensorflow::functor::CleanupSegments<bool*, bool*, tensorflow::functor::And>(bool*, bool*, int, int, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type)* 28�@�H�bAll_64huMUB
�
�void tensorflow::functor::CleanupSegments<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)* 28�@�H�b conv2d_87/kernel/Regularizer/SumhuMUB
�
�void tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)�*  2@8�@�H�b2gradient_tape/model_3/dense_10/BiasAdd/BiasAddGradhuZU�B
�
�void tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)�*  2@8�@�H�b1gradient_tape/model_3/dense_9/BiasAdd/BiasAddGradhuZU�B
�
�void tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)�!*  2@8�@�H�bBgradient_tape/model_3/batch_normalization_38/batchnorm/add_1/Sum_1huZU�B
�
�void tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)�!*  2@8�@�H�bBgradient_tape/model_3/batch_normalization_39/batchnorm/add_1/Sum_1huZU�B
�
�void tensorflow::functor::ColumnReduceKernel<float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, tensorflow::functor::Sum<float> >(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)�!*  2@8�@�H�b+model_3/batch_normalization_38/moments/meanhuZU�B
�
�void tensorflow::functor::ColumnReduceKernel<float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, tensorflow::functor::Sum<float> >(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)�!*  2@8�@�H�b/model_3/batch_normalization_38/moments/variancehuZU�B
�
�void tensorflow::functor::ColumnReduceKernel<float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, tensorflow::functor::Sum<float> >(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)�!*  2@8�@�H�b+model_3/batch_normalization_39/moments/meanhuZU�B
�
�void tensorflow::functor::ColumnReduceKernel<float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, tensorflow::functor::Sum<float> >(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)�!*  2@8�@�H�b/model_3/batch_normalization_39/moments/variancehuZU�B
�
�void tensorflow::functor::RowReduceKernel<Eigen::half const*, Eigen::half*, cub::Max>(Eigen::half const*, Eigen::half*, int, int, cub::Max, std::iterator_traits<Eigen::half const*>::value_type)*�28�@�H�bmodel_3/activation_43/Softmaxhu  �B
�
�void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*�28�@�H�Xbmodel_3/conv2d_76/Conv2DhuZU�B
�
�void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*�28�@�H�Xb;gradient_tape/model_3/conv2d_77/Conv2D/Conv2DBackpropFilterhuZU�B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�28�@�H�bIsFinite_29hu  �B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bmul_31huZU�B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bmul_60huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�b.model_3/batch_normalization_39/AssignMovingAvghuZU�B
�
�void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float>, float>(float*, float*, int, tensorflow::functor::Sum<float>, float)0*�28�@�H�b conv2d_88/kernel/Regularizer/Sumhu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�@�H�b;cond_1/then/_10/cond_1/Adam/Adam/update_3/ResourceApplyAdamhuZU�B
�
�void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *�28�@�H�bAll_45hu  �B
�
�void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *�28�@�H�bAll_49hu  �B
�
�void tensorflow::functor::CleanupSegments<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)* 28�@�H�b conv2d_84/kernel/Regularizer/SumhuMUB
�
�void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*�28�@�H�Xb;gradient_tape/model_3/conv2d_76/Conv2D/Conv2DBackpropFilterhuZU�B
o
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b,model_3/batch_normalization_39/batchnorm/addhuZU�B
X
"AddV2_GPU_DT_INT64_DT_INT64_kernel*�28�@�H�bcond/then/_0/cond/addhuZU�B
b
"AddV2_GPU_DT_INT64_DT_INT64_kernel*�28�@�H�bcond_1/then/_10/cond_1/Adam/addhuZU�B

 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*�2@8�@�H�b>gradient_tape/model_3/batch_normalization_38/moments/truediv_1huZU�B
}
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*�2@8�@�H�b<gradient_tape/model_3/batch_normalization_39/moments/truedivhuZU�B
G
!Equal_GPU_DT_INT64_DT_BOOL_kernel*�28�@�H�bEqualhuZU�B
o
'GreaterEqual_GPU_DT_HALF_DT_BOOL_kernel*�28�@�H�b'model_3/dropout_22/dropout/GreaterEqualhuZU�B
o
'GreaterEqual_GPU_DT_HALF_DT_BOOL_kernel*�28�@�H�b'model_3/dropout_23/dropout/GreaterEqualhuZU�B
g
(GreaterEqual_GPU_DT_INT64_DT_BOOL_kernel*�28�@�H�bcond/then/_0/cond/GreaterEqualhuZU�B
M
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�28�@�H�bIsFinitehu  �B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�28�@�H�bIsFinite_11hu  �B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�28�@�H�bIsFinite_13hu  �B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�28�@�H�bIsFinite_14hu  �B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�28�@�H�bIsFinite_17hu  �B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�28�@�H�b
IsFinite_2hu  �B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�28�@�H�bIsFinite_22hu  �B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�28�@�H�bIsFinite_23hu  �B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�28�@�H�bIsFinite_25hu  �B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�28�@�H�bIsFinite_27hu  �B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�28�@�H�b
IsFinite_3hu  �B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�28�@�H�bIsFinite_30hu  �B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�28�@�H�bIsFinite_33hu  �B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�28�@�H�bIsFinite_35hu  �B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�28�@�H�bIsFinite_39hu  �B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�28�@�H�bIsFinite_41hu  �B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�28�@�H�bIsFinite_43hu  �B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�28�@�H�bIsFinite_45hu  �B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�28�@�H�bIsFinite_49hu  �B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�28�@�H�b
IsFinite_5hu  �B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�28�@�H�bIsFinite_51hu  �B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�28�@�H�bIsFinite_53hu  �B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�28�@�H�b
IsFinite_6hu  �B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�28�@�H�bIsFinite_65hu  �B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�28�@�H�b
IsFinite_7hu  �B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�28�@�H�bIsFinite_46hu  �B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�28�@�H�bIsFinite_57hu  �B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�28�@�H�bIsFinite_58hu  �B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�28�@�H�bIsFinite_62hu  �B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�28�@�H�bIsFinite_63hu  �B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�28�@�H�bIsFinite_54hu  �B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�28�@�H�b
IsFinite_4hu  �B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�28�@�H�bIsFinite_64hu  �B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*�28�@�H�b
IsFinite_8hu  �B
P
%LogicalAnd_GPU_DT_BOOL_DT_BOOL_kernel*�28�@�H�b
LogicalAndhuZU�B
�
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bLgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mulhuZU�B
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b conv2d_75/kernel/Regularizer/mulhuZU�B
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b conv2d_78/kernel/Regularizer/mulhuZU�B
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b conv2d_79/kernel/Regularizer/mulhuZU�B
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b conv2d_80/kernel/Regularizer/mulhuZU�B
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b conv2d_81/kernel/Regularizer/mulhuZU�B
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b conv2d_82/kernel/Regularizer/mulhuZU�B
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b conv2d_84/kernel/Regularizer/mulhuZU�B
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b conv2d_87/kernel/Regularizer/mulhuZU�B
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b conv2d_88/kernel/Regularizer/mulhuZU�B
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b conv2d_89/kernel/Regularizer/mulhuZU�B
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b conv2d_90/kernel/Regularizer/mulhuZU�B
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b conv2d_91/kernel/Regularizer/mulhuZU�B
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b conv2d_93/kernel/Regularizer/mulhuZU�B
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b conv2d_94/kernel/Regularizer/mulhuZU�B
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b conv2d_95/kernel/Regularizer/mulhuZU�B
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bdense_10/kernel/Regularizer/mulhuZU�B
_
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bdense_9/kernel/Regularizer/mulhuZU�B
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b.gradient_tape/conv2d_75/kernel/Regularizer/MulhuZU�B
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b4gradient_tape/conv2d_75/kernel/Regularizer/mul/Mul_1huZU�B
�
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b@gradient_tape/model_3/batch_normalization_39/batchnorm/mul/Mul_1huZU�B
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b4model_3/batch_normalization_38/AssignMovingAvg_1/mulhuZU�B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b2model_3/batch_normalization_39/AssignMovingAvg/mulhuZU�B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bmul_13huZU�B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bmul_17huZU�B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bmul_19huZU�B
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bmul_2huZU�B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bmul_25huZU�B
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bmul_3huZU�B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bmul_32huZU�B
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bmul_4huZU�B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bmul_41huZU�B
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bmul_5huZU�B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bmul_51huZU�B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bmul_53huZU�B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bmul_57huZU�B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bmul_61huZU�B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bmul_65huZU�B
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bmul_7huZU�B
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bmul_9huZU�B
q
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b0gradient_tape/conv2d_76/kernel/Regularizer/Mul_1huZU�B

 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b>gradient_tape/model_3/batch_normalization_38/batchnorm/mul/MulhuZU�B
�
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bBgradient_tape/model_3/batch_normalization_38/batchnorm/mul_2/Mul_1huZU�B

 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b>gradient_tape/model_3/batch_normalization_39/batchnorm/mul/MulhuZU�B
�
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b@gradient_tape/model_3/batch_normalization_39/batchnorm/mul_2/MulhuZU�B
�
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bBgradient_tape/model_3/batch_normalization_39/batchnorm/mul_2/Mul_1huZU�B
m
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b,model_3/batch_normalization_39/batchnorm/mulhuZU�B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bmul_64huZU�B
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b.gradient_tape/conv2d_77/kernel/Regularizer/MulhuZU�B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bmul_66huZU�B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bmul_10huZU�B
_
Mul_GPU_DT_HALF_DT_HALF_kernel*�28�@�H�b model_3/dropout_22/dropout/Mul_1huZU�B
]
Mul_GPU_DT_HALF_DT_HALF_kernel*�28�@�H�bmodel_3/dropout_23/dropout/MulhuZU�B
b
 Pow_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b!cond_1/then/_10/cond_1/Adam/Pow_1huZU�B
q
"Rsqrt_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b.model_3/batch_normalization_39/batchnorm/Rsqrthu  �B
g
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b#conv2d_76/kernel/Regularizer/SquarehuZU�B
g
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b#conv2d_77/kernel/Regularizer/SquarehuZU�B
f
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b"dense_11/kernel/Regularizer/SquarehuZU�B
g
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b#conv2d_78/kernel/Regularizer/SquarehuZU�B
u
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b4model_3/batch_normalization_38/AssignMovingAvg_1/subhuZU�B
m
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b,model_3/batch_normalization_38/batchnorm/subhuZU�B
u
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b4model_3/batch_normalization_39/AssignMovingAvg_1/subhuZU�B
m
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b,model_3/batch_normalization_39/batchnorm/subhuZU�B
y
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�2@8�@�H�b8gradient_tape/model_3/batch_normalization_39/moments/subhuZU�B
�
htensorflow::functor::ReluGradHalfKernelVector(Eigen::half const*, Eigen::half const*, Eigen::half*, int)*�28�@�H�b,gradient_tape/model_3/activation_42/ReluGradhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2@8�@�H�bmodel_3/dropout_23/dropout/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�b-categorical_crossentropy/weighted_loss/Cast_1huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bUgradient_tape/Cast_3/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bzgradient_tape/categorical_crossentropy/weighted_loss/Cast/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bmodel_3/conv2d_75/BiasAdd/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bmodel_3/conv2d_75/Conv2D/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bmodel_3/conv2d_76/Conv2D/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bmodel_3/conv2d_79/BiasAdd/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bmodel_3/conv2d_81/BiasAdd/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bmodel_3/conv2d_82/BiasAdd/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bmodel_3/conv2d_83/BiasAdd/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bmodel_3/conv2d_85/BiasAdd/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bmodel_3/conv2d_86/BiasAdd/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bmodel_3/conv2d_87/BiasAdd/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bmodel_3/conv2d_89/BiasAdd/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bmodel_3/conv2d_90/BiasAdd/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bmodel_3/conv2d_92/BiasAdd/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bmodel_3/conv2d_94/BiasAdd/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bmodel_3/dense_11/BiasAdd/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bmodel_3/dense_10/BiasAdd/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bmodel_3/dense_9/BiasAdd/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bmodel_3/conv2d_77/Conv2D/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bmodel_3/dense_11/MatMul/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2@8�@�H�b6gradient_tape/model_3/batch_normalization_38/Cast/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2P8�@�H�bmodel_3/conv2d_87/Conv2D/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�b,categorical_crossentropy/weighted_loss/valuehuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bdiv_no_nan_1huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_rsqrt_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_rsqrt_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�b@gradient_tape/model_3/batch_normalization_38/batchnorm/RsqrtGradhuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_rsqrt_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_rsqrt_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�b@gradient_tape/model_3/batch_normalization_39/batchnorm/RsqrtGradhuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bCast_1huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bCast_6huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�b}categorical_crossentropy/softmax_cross_entropy_with_logits/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Cast_1huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�b?categorical_crossentropy/softmax_cross_entropy_with_logits/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bAcategorical_crossentropy/softmax_cross_entropy_with_logits/Cast_1huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bgcategorical_crossentropy/weighted_loss/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�b�gradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/Cast_2/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�b@gradient_tape/categorical_crossentropy/weighted_loss/Cast_1/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�b0gradient_tape/model_3/conv2d_75/Conv2D/Cast/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�b1gradient_tape/model_3/conv2d_76/BiasAdd/Cast/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�b0gradient_tape/model_3/conv2d_76/Conv2D/Cast/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�b1gradient_tape/model_3/conv2d_77/BiasAdd/Cast/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�b1gradient_tape/model_3/conv2d_79/BiasAdd/Cast/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�b1gradient_tape/model_3/conv2d_80/BiasAdd/Cast/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�b1gradient_tape/model_3/conv2d_83/BiasAdd/Cast/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�b1gradient_tape/model_3/conv2d_84/BiasAdd/Cast/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�b1gradient_tape/model_3/conv2d_85/BiasAdd/Cast/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�b1gradient_tape/model_3/conv2d_86/BiasAdd/Cast/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�b1gradient_tape/model_3/conv2d_87/BiasAdd/Cast/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�b1gradient_tape/model_3/conv2d_89/BiasAdd/Cast/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�b1gradient_tape/model_3/conv2d_90/BiasAdd/Cast/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�b1gradient_tape/model_3/conv2d_92/BiasAdd/Cast/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�b1gradient_tape/model_3/conv2d_93/BiasAdd/Cast/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�b1gradient_tape/model_3/conv2d_94/BiasAdd/Cast/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�b0gradient_tape/model_3/dense_11/BiasAdd/Cast/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�b0gradient_tape/model_3/dense_10/BiasAdd/Cast/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�b/gradient_tape/model_3/dense_9/BiasAdd/Cast/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�b/gradient_tape/model_3/dense_11/MatMul/Cast/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�b0gradient_tape/model_3/conv2d_78/Conv2D/Cast/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2@8�@�H�b8gradient_tape/model_3/batch_normalization_39/Cast_1/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2@8�@�H�b0gradient_tape/model_3/conv2d_81/Conv2D/Cast/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2@8�@�H�b#model_3/batch_normalization_38/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bCast_2huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bCast_8huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�b;gradient_tape/model_3/batch_normalization_38/moments/Cast_1huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�b9gradient_tape/model_3/batch_normalization_39/moments/CasthuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�b;gradient_tape/model_3/batch_normalization_39/moments/Cast_1huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�b0model_3/batch_normalization_38/AssignMovingAvg_1huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�b0model_3/batch_normalization_39/AssignMovingAvg_1huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bAdam/gradients/AddN_10huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bAdam/gradients/AddNhuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bAssignAddVariableOp_1huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�bAssignAddVariableOp_3huZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�@�H�b4cond_1/then/_10/cond_1/Adam/Adam/AssignAddVariableOphuZU�B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorEvalToOp<Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_log_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorEvalToOp<Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_log_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>, long)*�28�@�H�b:categorical_crossentropy/softmax_cross_entropy_with_logitshuZU�B
�
�void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *�28�@�H�bAll_10hu���B
�
�void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *�28�@�H�bAll_12hu���B
�
�void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *�28�@�H�bAll_26hu���B