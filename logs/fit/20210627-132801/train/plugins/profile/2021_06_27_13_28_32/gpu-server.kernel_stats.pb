
’
Ïvoid cudnn::cnn::wgrad_alg1_engine_NHWC<__half, float, 128, 6, 8, 3, 3, 5, false, true>(int, int, int, __half const*, int, float*, __half const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int){ÑR* 2¥8ä∫î	@ä∫î	Hä∫î	Xb<gradient_tape/model_7/conv2d_190/Conv2D/Conv2DBackpropFilterhuMUB
˚
ï_ZN13cutlass_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi64ELi256ELi32EEENS4_51Conv2dWgradOutputGradientTileAccessIteratorAnalyticINS_11MatrixShapeILi64ELi32EEENS_6half_tENS_9transform29PitchLinearWarpRakedThreadMapINS_6layout16PitchLinearShapeILi64ELi32EEELi128ENSG_ILi8ELi4EEELi8EEEEENSD_11threadblock25RegularTileAccessIteratorISB_SC_NSF_40ColumnMajorTensorOpMultiplicandCongruousILi16ELi64EEELi1ESJ_Li16EEELNS_4arch14CacheOperation4KindE0ENS4_47Conv2dWgradActivationTileAccessIteratorAnalyticINSA_ILi32ELi256EEESC_NSE_INSG_ILi256ELi32EEELi128ESI_Li8EEEEENSM_ISU_SC_NSF_37RowMajorTensorOpMultiplicandCongruousILi16ELi64EEELi0ESW_Li16EEELSS_0ENS6_11threadblock9MmaPolicyINS6_4warp11MmaTensorOpINS7_ILi64ELi64ELi32EEESC_SO_SC_SZ_fNSF_8RowMajorENS13_17MmaTensorOpPolicyINSQ_3MmaINS7_ILi16ELi8ELi16EEELi32ESC_S16_SC_NSF_11ColumnMajorEfS16_NSQ_13OpMultiplyAddEEENSA_ILi1ELi1EEEEELi1ELb0EbEENSA_ILi0ELi0EEES1G_Li1EEELi4EbEENS_8epilogue11threadblock8EpilogueIS8_S1F_Li1ENS1K_22PredicatedTileIteratorINS1K_26OutputTileOptimalThreadMapINS1K_15OutputTileShapeILi256ELi8ELi1ELi1ELi1EEENS1O_ILi1ELi8ELi1ELi1ELi8EEELi128ELi4ELi32EEEfEENS1J_4warp24FragmentIteratorTensorOpIS15_S19_fNS_5ArrayIfLi4ELb1EEES16_EENS1T_20TileIteratorTensorOpIS15_S19_fS16_EENS1K_18SharedLoadIteratorINS1R_18CompactedThreadMapEfLi16EEENS1J_6thread17LinearCombinationIfLi4EffLNS_15FloatRoundStyleE2EEENSA_ILi0ELi8EEEEENS11_30GemmIdentityThreadblockSwizzleILi4EEELNS1_8OperatorE2ENS1_17Conv2dProblemSizeEEEEEvNT_6ParamsEˇ ÄÄ*Ä22)8∏±®@∏±®H∏±®Xb<gradient_tape/model_7/conv2d_179/Conv2D/Conv2DBackpropFilterh
ú
∂void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride>(cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride::Params)ı Ä¿*Ä2Ä8°…∆@°…∆H°…∆Xb;gradient_tape/model_7/conv2d_179/Conv2D/Conv2DBackpropInputh
‡
úvoid cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3>(cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3::Params)˛ Ä¿*Ä2Ä8ûÒ∑@ûÒ∑HûÒ∑Xbmodel_7/conv2d_179/Conv2Dh
ô
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8òŸü@òŸüHòŸüb<cond_1/then/_10/cond_1/Adam/Adam/update_56/ResourceApplyAdamhuZUÖB
∆
ﬁvoid xmma_cudnn::gemm::kernel<xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, xmma_cudnn::Col, 256, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 3> >(xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, xmma_cudnn::Col, 256, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 3>::Params)Ó Ä¿*Ä2$	8˙¯ñ@˙¯ñH˙¯ñPXb<gradient_tape/model_7/conv2d_178/Conv2D/Conv2DBackpropFilterh
ú
∂void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride>(cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride::Params)ı Ä¿*Ä2Ä8ˆËÜ@ˆËÜHˆËÜXb;gradient_tape/model_7/conv2d_178/Conv2D/Conv2DBackpropInputh
‡
úvoid cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3>(cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3::Params)˛ Ä¿*Ä2Ä8ËÏ@ËÏHËÏXbmodel_7/conv2d_178/Conv2Dh
˛
ò_ZN13cutlass_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi128ELi256ELi32EEENS4_51Conv2dWgradOutputGradientTileAccessIteratorAnalyticINS_11MatrixShapeILi128ELi32EEENS_6half_tENS_9transform29PitchLinearWarpRakedThreadMapINS_6layout16PitchLinearShapeILi128ELi32EEELi256ENSG_ILi8ELi4EEELi8EEEEENSD_11threadblock25RegularTileAccessIteratorISB_SC_NSF_40ColumnMajorTensorOpMultiplicandCongruousILi16ELi64EEELi1ESJ_Li16EEELNS_4arch14CacheOperation4KindE0ENS4_47Conv2dWgradActivationTileAccessIteratorAnalyticINSA_ILi32ELi256EEESC_NSE_INSG_ILi256ELi32EEELi256ESI_Li8EEEEENSM_ISU_SC_NSF_37RowMajorTensorOpMultiplicandCongruousILi16ELi64EEELi0ESW_Li16EEELSS_0ENS6_11threadblock9MmaPolicyINS6_4warp11MmaTensorOpINS7_ILi64ELi64ELi32EEESC_SO_SC_SZ_fNSF_8RowMajorENS13_17MmaTensorOpPolicyINSQ_3MmaINS7_ILi16ELi8ELi16EEELi32ESC_S16_SC_NSF_11ColumnMajorEfS16_NSQ_13OpMultiplyAddEEENSA_ILi1ELi1EEEEELi1ELb0EbEENSA_ILi0ELi0EEES1G_Li1EEELi3EbEENS_8epilogue11threadblock8EpilogueIS8_S1F_Li1ENS1K_22PredicatedTileIteratorINS1K_26OutputTileOptimalThreadMapINS1K_15OutputTileShapeILi256ELi8ELi2ELi1ELi1EEENS1O_ILi1ELi8ELi1ELi1ELi8EEELi256ELi4ELi32EEEfEENS1J_4warp24FragmentIteratorTensorOpIS15_S19_fNS_5ArrayIfLi4ELb1EEES16_EENS1T_20TileIteratorTensorOpIS15_S19_fS16_EENS1K_18SharedLoadIteratorINS1R_18CompactedThreadMapEfLi16EEENS1J_6thread17LinearCombinationIfLi4EffLNS_15FloatRoundStyleE2EEENSA_ILi0ELi8EEEEENS11_30GemmIdentityThreadblockSwizzleILi4EEELNS1_8OperatorE2ENS1_17Conv2dProblemSizeEEEEEvNT_6ParamsEˇ Ä¿*Ä2?8Ô¯Ë@Ô¯ËHÔ¯ËXb<gradient_tape/model_7/conv2d_185/Conv2D/Conv2DBackpropFilterh
∂
Œvoid xmma_cudnn::gemm::kernel<xmma_cudnn::implicit_gemm::dgrad::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_a_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 16, false, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 16, xmma_cudnn::Row, 32, 256> >, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_c_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, 16, xmma_cudnn::Fragment_c<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, false>, false>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 3> >(xmma_cudnn::implicit_gemm::dgrad::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_a_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 16, false, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 16, xmma_cudnn::Row, 32, 256> >, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_c_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, 16, xmma_cudnn::Fragment_c<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, false>, false>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 3>::Params)ˇ Ä‡*Ä2
Ä8ÌÄ·@ÌÄ·HÌÄ·PXb;gradient_tape/model_7/conv2d_185/Conv2D/Conv2DBackpropInputh
‡
úvoid cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3>(cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3::Params)˛ Ä¿*Ä2Ä8Ïÿ€@Ïÿ€HÏÿ€Xbmodel_7/conv2d_185/Conv2Dh
˛
ò_ZN13cutlass_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi128ELi256ELi32EEENS4_51Conv2dWgradOutputGradientTileAccessIteratorAnalyticINS_11MatrixShapeILi128ELi32EEENS_6half_tENS_9transform29PitchLinearWarpRakedThreadMapINS_6layout16PitchLinearShapeILi128ELi32EEELi256ENSG_ILi8ELi4EEELi8EEEEENSD_11threadblock25RegularTileAccessIteratorISB_SC_NSF_40ColumnMajorTensorOpMultiplicandCongruousILi16ELi64EEELi1ESJ_Li16EEELNS_4arch14CacheOperation4KindE0ENS4_47Conv2dWgradActivationTileAccessIteratorAnalyticINSA_ILi32ELi256EEESC_NSE_INSG_ILi256ELi32EEELi256ESI_Li8EEEEENSM_ISU_SC_NSF_37RowMajorTensorOpMultiplicandCongruousILi16ELi64EEELi0ESW_Li16EEELSS_0ENS6_11threadblock9MmaPolicyINS6_4warp11MmaTensorOpINS7_ILi64ELi64ELi32EEESC_SO_SC_SZ_fNSF_8RowMajorENS13_17MmaTensorOpPolicyINSQ_3MmaINS7_ILi16ELi8ELi16EEELi32ESC_S16_SC_NSF_11ColumnMajorEfS16_NSQ_13OpMultiplyAddEEENSA_ILi1ELi1EEEEELi1ELb0EbEENSA_ILi0ELi0EEES1G_Li1EEELi3EbEENS_8epilogue11threadblock8EpilogueIS8_S1F_Li1ENS1K_22PredicatedTileIteratorINS1K_26OutputTileOptimalThreadMapINS1K_15OutputTileShapeILi256ELi8ELi2ELi1ELi1EEENS1O_ILi1ELi8ELi1ELi1ELi8EEELi256ELi4ELi32EEEfEENS1J_4warp24FragmentIteratorTensorOpIS15_S19_fNS_5ArrayIfLi4ELb1EEES16_EENS1T_20TileIteratorTensorOpIS15_S19_fS16_EENS1K_18SharedLoadIteratorINS1R_18CompactedThreadMapEfLi16EEENS1J_6thread17LinearCombinationIfLi4EffLNS_15FloatRoundStyleE2EEENSA_ILi0ELi8EEEEENS11_30GemmIdentityThreadblockSwizzleILi4EEELNS1_8OperatorE2ENS1_17Conv2dProblemSizeEEEEEvNT_6ParamsEˇ Ä¿*Ä228ÿ¿Ñ@ÿ¿ÑHÿ¿ÑXb<gradient_tape/model_7/conv2d_182/Conv2D/Conv2DBackpropFilterh
‡
úvoid cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3>(cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3::Params)˛ Ä¿*Ä2Ä8÷ÿ˘@÷ÿ˘H÷ÿ˘Xbmodel_7/conv2d_182/Conv2Dh
∂
Œvoid xmma_cudnn::gemm::kernel<xmma_cudnn::implicit_gemm::dgrad::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_a_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 16, false, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 16, xmma_cudnn::Row, 32, 256> >, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_c_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, 16, xmma_cudnn::Fragment_c<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, false>, false>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 3> >(xmma_cudnn::implicit_gemm::dgrad::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_a_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 16, false, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 16, xmma_cudnn::Row, 32, 256> >, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_c_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, 16, xmma_cudnn::Fragment_c<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, false>, false>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 3>::Params)ˇ Ä‡*Ä2Ä8‘¯Ò@‘¯ÒH‘¯ÒPXb;gradient_tape/model_7/conv2d_182/Conv2D/Conv2DBackpropInputh
¥
Œvoid cudnn::ops::pooling_bw_kernel_max<__half, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, false>(cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor) Ä *Ä2Ä@8‘êÓ@‘êÓH‘êÓb:gradient_tape/model_7/max_pooling2d_29/MaxPool/MaxPoolGradhu  »B
‡
úvoid cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3>(cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3::Params)˛ Ä¿*Ä2Ä8Õò‘@Õò‘HÕò‘Xbmodel_7/conv2d_190/Conv2Dh
≥
Œvoid cudnn::ops::pooling_bw_kernel_max<__half, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, false>(cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor) Ä *Ä2  @8Ã∞œ@Ã∞œHÃ∞œb:gradient_tape/model_7/max_pooling2d_28/MaxPool/MaxPoolGradhu  »B
ú
∂void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride>(cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride::Params)ı Ä¿*Ä2Ä8«»∑@«»∑H«»∑Xb;gradient_tape/model_7/conv2d_190/Conv2D/Conv2DBackpropInputh
˚
ï_ZN13cutlass_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi64ELi256ELi32EEENS4_51Conv2dWgradOutputGradientTileAccessIteratorAnalyticINS_11MatrixShapeILi64ELi32EEENS_6half_tENS_9transform29PitchLinearWarpRakedThreadMapINS_6layout16PitchLinearShapeILi64ELi32EEELi128ENSG_ILi8ELi4EEELi8EEEEENSD_11threadblock25RegularTileAccessIteratorISB_SC_NSF_40ColumnMajorTensorOpMultiplicandCongruousILi16ELi64EEELi1ESJ_Li16EEELNS_4arch14CacheOperation4KindE0ENS4_47Conv2dWgradActivationTileAccessIteratorAnalyticINSA_ILi32ELi256EEESC_NSE_INSG_ILi256ELi32EEELi128ESI_Li8EEEEENSM_ISU_SC_NSF_37RowMajorTensorOpMultiplicandCongruousILi16ELi64EEELi0ESW_Li16EEELSS_0ENS6_11threadblock9MmaPolicyINS6_4warp11MmaTensorOpINS7_ILi64ELi64ELi32EEESC_SO_SC_SZ_fNSF_8RowMajorENS13_17MmaTensorOpPolicyINSQ_3MmaINS7_ILi16ELi8ELi16EEELi32ESC_S16_SC_NSF_11ColumnMajorEfS16_NSQ_13OpMultiplyAddEEENSA_ILi1ELi1EEEEELi1ELb0EbEENSA_ILi0ELi0EEES1G_Li1EEELi4EbEENS_8epilogue11threadblock8EpilogueIS8_S1F_Li1ENS1K_22PredicatedTileIteratorINS1K_26OutputTileOptimalThreadMapINS1K_15OutputTileShapeILi256ELi8ELi1ELi1ELi1EEENS1O_ILi1ELi8ELi1ELi1ELi8EEELi128ELi4ELi32EEEfEENS1J_4warp24FragmentIteratorTensorOpIS15_S19_fNS_5ArrayIfLi4ELb1EEES16_EENS1T_20TileIteratorTensorOpIS15_S19_fS16_EENS1K_18SharedLoadIteratorINS1R_18CompactedThreadMapEfLi16EEENS1J_6thread17LinearCombinationIfLi4EffLNS_15FloatRoundStyleE2EEENSA_ILi0ELi8EEEEENS11_30GemmIdentityThreadblockSwizzleILi4EEELNS1_8OperatorE2ENS1_17Conv2dProblemSizeEEEEEvNT_6ParamsEˇ ÄÄ*Ä2 8¡ù@¡ùH¡ùXb<gradient_tape/model_7/conv2d_191/Conv2D/Conv2DBackpropFilterh
≠
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ω–ç@Ω–çHΩ–çbAdam/gradients/AddN_31huZUÖB
x
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ä¿8ª–Ñ@ª–ÑHª–Ñb/gradient_tape/dense_21/kernel/Regularizer/Mul_1huZUÖB
ﬁ
övoid cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816fprop_optimized_f16_64x128_32x6>(cutlass_tensorop_f16_s16816fprop_optimized_f16_64x128_32x6::Params)® Ä¿*Ä2Ä8∑àÔ@∑àÔH∑àÔXbmodel_7/conv2d_191/Conv2Dh
ú
∂void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride>(cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride::Params)ı Ä¿*Ä2Ä8≥®‡@≥®‡H≥®‡Xb;gradient_tape/model_7/conv2d_191/Conv2D/Conv2DBackpropInputh
∆
ﬁvoid xmma_cudnn::gemm::kernel<xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 32, 2, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 32, 2, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 32, 2, 2, 1, 1>, 16, xmma_cudnn::Col, 128, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 32, 2, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 32, 2, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 4> >(xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 32, 2, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 32, 2, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 32, 2, 2, 1, 1>, 16, xmma_cudnn::Col, 128, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 32, 2, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 128, 128, 32, 2, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 4>::Params)Ù ÄÄ*Ä2	8ØÄ“@ØÄ“HØÄ“PXb<gradient_tape/model_7/conv2d_176/Conv2D/Conv2DBackpropFilterh
ú
∂void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride>(cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride::Params)ı Ä¿*Ä2Ä8Ø‡Ã@Ø‡ÃHØ‡ÃXb;gradient_tape/model_7/conv2d_184/Conv2D/Conv2DBackpropInputh
˛
ò_ZN13cutlass_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi128ELi256ELi32EEENS4_51Conv2dWgradOutputGradientTileAccessIteratorAnalyticINS_11MatrixShapeILi128ELi32EEENS_6half_tENS_9transform29PitchLinearWarpRakedThreadMapINS_6layout16PitchLinearShapeILi128ELi32EEELi256ENSG_ILi8ELi4EEELi8EEEEENSD_11threadblock25RegularTileAccessIteratorISB_SC_NSF_40ColumnMajorTensorOpMultiplicandCongruousILi16ELi64EEELi1ESJ_Li16EEELNS_4arch14CacheOperation4KindE0ENS4_47Conv2dWgradActivationTileAccessIteratorAnalyticINSA_ILi32ELi256EEESC_NSE_INSG_ILi256ELi32EEELi256ESI_Li8EEEEENSM_ISU_SC_NSF_37RowMajorTensorOpMultiplicandCongruousILi16ELi64EEELi0ESW_Li16EEELSS_0ENS6_11threadblock9MmaPolicyINS6_4warp11MmaTensorOpINS7_ILi64ELi64ELi32EEESC_SO_SC_SZ_fNSF_8RowMajorENS13_17MmaTensorOpPolicyINSQ_3MmaINS7_ILi16ELi8ELi16EEELi32ESC_S16_SC_NSF_11ColumnMajorEfS16_NSQ_13OpMultiplyAddEEENSA_ILi1ELi1EEEEELi1ELb0EbEENSA_ILi0ELi0EEES1G_Li1EEELi3EbEENS_8epilogue11threadblock8EpilogueIS8_S1F_Li1ENS1K_22PredicatedTileIteratorINS1K_26OutputTileOptimalThreadMapINS1K_15OutputTileShapeILi256ELi8ELi2ELi1ELi1EEENS1O_ILi1ELi8ELi1ELi1ELi8EEELi256ELi4ELi32EEEfEENS1J_4warp24FragmentIteratorTensorOpIS15_S19_fNS_5ArrayIfLi4ELb1EEES16_EENS1T_20TileIteratorTensorOpIS15_S19_fS16_EENS1K_18SharedLoadIteratorINS1R_18CompactedThreadMapEfLi16EEENS1J_6thread17LinearCombinationIfLi4EffLNS_15FloatRoundStyleE2EEENSA_ILi0ELi8EEEEENS11_30GemmIdentityThreadblockSwizzleILi4EEELNS1_8OperatorE2ENS1_17Conv2dProblemSizeEEEEEvNT_6ParamsEˇ Ä¿*Ä28´¿ø@´¿øH´¿øXb<gradient_tape/model_7/conv2d_184/Conv2D/Conv2DBackpropFilterh
∂
Œvoid xmma_cudnn::gemm::kernel<xmma_cudnn::implicit_gemm::dgrad::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_a_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 16, false, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 16, xmma_cudnn::Row, 32, 256> >, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_c_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, 16, xmma_cudnn::Fragment_c<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, false>, false>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 3> >(xmma_cudnn::implicit_gemm::dgrad::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_a_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 16, false, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 16, xmma_cudnn::Row, 32, 256> >, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_c_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, 16, xmma_cudnn::Fragment_c<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, false>, false>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 3>::Params)ˇ Ä‡*Ä2Ä8©†∏@©†∏H©†∏PXb;gradient_tape/model_7/conv2d_176/Conv2D/Conv2DBackpropInputh
‡
úvoid cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3>(cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3::Params)˛ Ä¿*Ä2Ä8®∏≥@®∏≥H®∏≥Xbmodel_7/conv2d_176/Conv2Dh
O
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ä¿8ßÿÆ@ßÿÆHßÿÆbmul_58huZUÖB
v
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ä¿8ß»Æ@ß»ÆHß»Æb-gradient_tape/dense_21/kernel/Regularizer/MulhuZUÖB
n
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ä¿8®¿Æ@®¿ÆH®¿Æb"dense_21/kernel/Regularizer/SquarehuZUÖB
‡
úvoid cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3>(cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3::Params)˛ Ä¿*Ä2Ä8®¯≠@®¯≠H®¯≠Xbmodel_7/conv2d_184/Conv2Dh
∫
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8ß¯¨@ß¯¨Hß¯¨b/gradient_tape/model_7/dense_21/MatMul/Cast/CasthuZUÖB
˚
ï_ZN13cutlass_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi64ELi256ELi32EEENS4_51Conv2dWgradOutputGradientTileAccessIteratorAnalyticINS_11MatrixShapeILi64ELi32EEENS_6half_tENS_9transform29PitchLinearWarpRakedThreadMapINS_6layout16PitchLinearShapeILi64ELi32EEELi128ENSG_ILi8ELi4EEELi8EEEEENSD_11threadblock25RegularTileAccessIteratorISB_SC_NSF_40ColumnMajorTensorOpMultiplicandCongruousILi16ELi64EEELi1ESJ_Li16EEELNS_4arch14CacheOperation4KindE0ENS4_47Conv2dWgradActivationTileAccessIteratorAnalyticINSA_ILi32ELi256EEESC_NSE_INSG_ILi256ELi32EEELi128ESI_Li8EEEEENSM_ISU_SC_NSF_37RowMajorTensorOpMultiplicandCongruousILi16ELi64EEELi0ESW_Li16EEELSS_0ENS6_11threadblock9MmaPolicyINS6_4warp11MmaTensorOpINS7_ILi64ELi64ELi32EEESC_SO_SC_SZ_fNSF_8RowMajorENS13_17MmaTensorOpPolicyINSQ_3MmaINS7_ILi16ELi8ELi16EEELi32ESC_S16_SC_NSF_11ColumnMajorEfS16_NSQ_13OpMultiplyAddEEENSA_ILi1ELi1EEEEELi1ELb0EbEENSA_ILi0ELi0EEES1G_Li1EEELi4EbEENS_8epilogue11threadblock8EpilogueIS8_S1F_Li1ENS1K_22PredicatedTileIteratorINS1K_26OutputTileOptimalThreadMapINS1K_15OutputTileShapeILi256ELi8ELi1ELi1ELi1EEENS1O_ILi1ELi8ELi1ELi1ELi8EEELi128ELi4ELi32EEEfEENS1J_4warp24FragmentIteratorTensorOpIS15_S19_fNS_5ArrayIfLi4ELb1EEES16_EENS1T_20TileIteratorTensorOpIS15_S19_fS16_EENS1K_18SharedLoadIteratorINS1R_18CompactedThreadMapEfLi16EEENS1J_6thread17LinearCombinationIfLi4EffLNS_15FloatRoundStyleE2EEENSA_ILi0ELi8EEEEENS11_30GemmIdentityThreadblockSwizzleILi4EEELNS1_8OperatorE2ENS1_17Conv2dProblemSizeEEEEEvNT_6ParamsEˇ ÄÄ*Ä28¶–£@¶–£H¶–£Xb<gradient_tape/model_7/conv2d_188/Conv2D/Conv2DBackpropFilterh
≥
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8£∞ö@£∞öH£∞öbmodel_7/dense_21/MatMul/CasthuZUÖB
ú
∂void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride>(cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride::Params)ı Ä¿*Ä2Ä8£àô@£àôH£àôXb;gradient_tape/model_7/conv2d_188/Conv2D/Conv2DBackpropInputh
˛
ò_ZN13cutlass_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi128ELi256ELi32EEENS4_51Conv2dWgradOutputGradientTileAccessIteratorAnalyticINS_11MatrixShapeILi128ELi32EEENS_6half_tENS_9transform29PitchLinearWarpRakedThreadMapINS_6layout16PitchLinearShapeILi128ELi32EEELi256ENSG_ILi8ELi4EEELi8EEEEENSD_11threadblock25RegularTileAccessIteratorISB_SC_NSF_40ColumnMajorTensorOpMultiplicandCongruousILi16ELi64EEELi1ESJ_Li16EEELNS_4arch14CacheOperation4KindE0ENS4_47Conv2dWgradActivationTileAccessIteratorAnalyticINSA_ILi32ELi256EEESC_NSE_INSG_ILi256ELi32EEELi256ESI_Li8EEEEENSM_ISU_SC_NSF_37RowMajorTensorOpMultiplicandCongruousILi16ELi64EEELi0ESW_Li16EEELSS_0ENS6_11threadblock9MmaPolicyINS6_4warp11MmaTensorOpINS7_ILi64ELi64ELi32EEESC_SO_SC_SZ_fNSF_8RowMajorENS13_17MmaTensorOpPolicyINSQ_3MmaINS7_ILi16ELi8ELi16EEELi32ESC_S16_SC_NSF_11ColumnMajorEfS16_NSQ_13OpMultiplyAddEEENSA_ILi1ELi1EEEEELi1ELb0EbEENSA_ILi0ELi0EEES1G_Li1EEELi3EbEENS_8epilogue11threadblock8EpilogueIS8_S1F_Li1ENS1K_22PredicatedTileIteratorINS1K_26OutputTileOptimalThreadMapINS1K_15OutputTileShapeILi256ELi8ELi2ELi1ELi1EEENS1O_ILi1ELi8ELi1ELi1ELi8EEELi256ELi4ELi32EEEfEENS1J_4warp24FragmentIteratorTensorOpIS15_S19_fNS_5ArrayIfLi4ELb1EEES16_EENS1T_20TileIteratorTensorOpIS15_S19_fS16_EENS1K_18SharedLoadIteratorINS1R_18CompactedThreadMapEfLi16EEENS1J_6thread17LinearCombinationIfLi4EffLNS_15FloatRoundStyleE2EEENSA_ILi0ELi8EEEEENS11_30GemmIdentityThreadblockSwizzleILi4EEELNS1_8OperatorE2ENS1_17Conv2dProblemSizeEEEEEvNT_6ParamsEˇ Ä¿*Ä28††é@††éH††éXb<gradient_tape/model_7/conv2d_181/Conv2D/Conv2DBackpropFilterh
‡
úvoid cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3>(cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3::Params)˛ Ä¿*Ä2Ä8†»ã@†»ãH†»ãXbmodel_7/conv2d_181/Conv2Dh
ú
∂void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride>(cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride::Params)ı Ä¿*Ä2Ä8†»â@†»âH†»âXb;gradient_tape/model_7/conv2d_181/Conv2D/Conv2DBackpropInputh
∂
Œvoid xmma_cudnn::gemm::kernel<xmma_cudnn::implicit_gemm::dgrad::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_a_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 16, false, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 16, xmma_cudnn::Row, 32, 256> >, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_c_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, 16, xmma_cudnn::Fragment_c<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, false>, false>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 3> >(xmma_cudnn::implicit_gemm::dgrad::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_a_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 16, false, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 16, xmma_cudnn::Row, 32, 256> >, xmma_cudnn::implicit_gemm::dgrad::Gmem_tile_c_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, 16, xmma_cudnn::Fragment_c<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, false>, false>, xmma_cudnn::implicit_gemm::Input_related<0, 0, 0, false>, 3>::Params)ˇ Ä‡*Ä2Ä8ûêÜ@ûêÜHûêÜPXb;gradient_tape/model_7/conv2d_175/Conv2D/Conv2DBackpropInputh
∆
ﬁvoid xmma_cudnn::gemm::kernel<xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, xmma_cudnn::Col, 256, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 3> >(xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, xmma_cudnn::Col, 256, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 3>::Params)Ó Ä¿*Ä2		8ûòÖ@ûòÖHûòÖPXb<gradient_tape/model_7/conv2d_175/Conv2D/Conv2DBackpropFilterh
‡
úvoid cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3>(cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3::Params)˛ Ä¿*Ä2Ä8û®Ñ@û®ÑHû®ÑXbmodel_7/conv2d_175/Conv2Dh
€
övoid cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816fprop_optimized_f16_64x128_32x6>(cutlass_tensorop_f16_s16816fprop_optimized_f16_64x128_32x6::Params)® Ä¿*Ä2Ä8õ®y@õ®yHõ®yXbmodel_7/conv2d_188/Conv2Dh
Ì
Évoid cudnn::bn_bw_1C11_kernel_new<__half, float, float2, 512, true, 1>(float, float, float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, float const*, float*, float*, float const*, float const*, float)(ê*Ä2Ä8õ–t@õ–tHõ–tbAgradient_tape/model_7/batch_normalization_73/FusedBatchNormGradV3hu  »B
ê
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2ÄÄ8õ∞t@â‡&HâË&bAgradient_tape/model_7/batch_normalization_72/FusedBatchNormGradV3hu  »B
±
Œvoid cudnn::ops::pooling_bw_kernel_max<__half, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, false>(cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor) Ä *Ä2†@8ö∞t@ö∞tHö∞tb:gradient_tape/model_7/max_pooling2d_30/MaxPool/MaxPoolGradhu  »B
Ì
Évoid cudnn::bn_bw_1C11_kernel_new<__half, float, float2, 512, true, 1>(float, float, float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, float const*, float*, float*, float const*, float const*, float)(ê*Ä2Ä8õ†t@õ†tHõ†tbAgradient_tape/model_7/batch_normalization_71/FusedBatchNormGradV3hu  »B
ê
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2ÄÄ8öòt@âÿ&Hâ‡&bAgradient_tape/model_7/batch_normalization_73/FusedBatchNormGradV3hu  »B
ê
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2ÄÄ8òs@à»&Hàÿ&bAgradient_tape/model_7/batch_normalization_71/FusedBatchNormGradV3hu  »B
Ì
Évoid cudnn::bn_bw_1C11_kernel_new<__half, float, float2, 512, true, 1>(float, float, float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, float const*, float*, float*, float const*, float const*, float)(ê*Ä2Ä8ö¯r@ö¯rHö¯rbAgradient_tape/model_7/batch_normalization_72/FusedBatchNormGradV3hu  »B
ô
∂void cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride>(cutlass_tensorop_f16_s16816dgrad_optimized_f16_128x256_32x3_unity_stride::Params)ı Ä¿*Ä2Ä8ôÄo@ôÄoHôÄoXb;gradient_tape/model_7/conv2d_187/Conv2D/Conv2DBackpropInputh
÷
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8ô∞j@ô∞jHô∞jb.gradient_tape/dense_21/kernel/Regularizer/TilehuZUÖB
U
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2ÄÄ8ò¿i@ò¿iHò¿ibIsFinite_56hu  »B
›
úvoid cutlass_cudnn::Kernel<cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3>(cutlass_tensorop_f16_s16816fprop_optimized_f16_256x128_32x3::Params)˛ Ä¿*Ä2Ä8ó‡b@ó‡bHó‡bXbmodel_7/conv2d_187/Conv2Dh
∑
“void xmma_cudnn::gemm::kernel<xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, 16, xmma_cudnn::Col, 256, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, false, 16, xmma_cudnn::Row, 64, 32> >, false, 3> >(xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, 16, xmma_cudnn::Col, 256, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 64, 32, 4, 1, 1, 1>, false, 16, xmma_cudnn::Row, 64, 32> >, false, 3>::Params)˛ Ä‡*Ä2Z8ï∏_@ï∏_Hï∏_PXb<gradient_tape/model_7/conv2d_187/Conv2D/Conv2DBackpropFilterh
±
Œvoid cudnn::ops::pooling_bw_kernel_max<__half, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, false>(cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor) Ä *Ä2†@8î∏T@î∏THî∏Tb:gradient_tape/model_7/max_pooling2d_31/MaxPool/MaxPoolGradhu  »B
¬
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2ú8ìËR@ìËRHìËRbdense_21/kernel/Regularizer/Sumhu  »B
˛
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2ÄÄ8ëÄN@à¯&Hâà'b/model_7/batch_normalization_72/FusedBatchNormV3hu  »B
˛
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2ÄÄ8í¯M@â¯&HâÄ'b/model_7/batch_normalization_73/FusedBatchNormV3hu  »B
˛
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2ÄÄ8ë»M@à‡&HâË&b/model_7/batch_normalization_71/FusedBatchNormV3hu  »B
“
Ïvoid cudnn::cnn::wgrad_alg1_engine_NHWC<__half, float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, __half const*, int, float*, __half const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)PÑ*2Ä8ë†M@ë†MHë†MXb<gradient_tape/model_7/conv2d_172/Conv2D/Conv2DBackpropFilterhu  HB
Ÿ

ü
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8ë¯K@ë¯KHë¯KbAdam/gradients/AddN_8huZUÖB
∞
ÿvoid cudnn::bn_fw_tr_1C11_kernel_NCHW<__half, float, 512, true, 1>(cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(ê*Ä2Ä8êÿE@êÿEHêÿEb/model_7/batch_normalization_72/FusedBatchNormV3hu  »B
∞
ÿvoid cudnn::bn_fw_tr_1C11_kernel_NCHW<__half, float, 512, true, 1>(cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(ê*Ä2Ä8ê∏E@ê∏EHê∏Eb/model_7/batch_normalization_73/FusedBatchNormV3hu  »B
∞
ÿvoid cudnn::bn_fw_tr_1C11_kernel_NCHW<__half, float, 512, true, 1>(cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(ê*Ä2Ä8è∏@@è∏@Hè∏@b/model_7/batch_normalization_71/FusedBatchNormV3hu  »B
¯
ï_ZN13cutlass_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi64ELi256ELi32EEENS4_51Conv2dWgradOutputGradientTileAccessIteratorAnalyticINS_11MatrixShapeILi64ELi32EEENS_6half_tENS_9transform29PitchLinearWarpRakedThreadMapINS_6layout16PitchLinearShapeILi64ELi32EEELi128ENSG_ILi8ELi4EEELi8EEEEENSD_11threadblock25RegularTileAccessIteratorISB_SC_NSF_40ColumnMajorTensorOpMultiplicandCongruousILi16ELi64EEELi1ESJ_Li16EEELNS_4arch14CacheOperation4KindE0ENS4_47Conv2dWgradActivationTileAccessIteratorAnalyticINSA_ILi32ELi256EEESC_NSE_INSG_ILi256ELi32EEELi128ESI_Li8EEEEENSM_ISU_SC_NSF_37RowMajorTensorOpMultiplicandCongruousILi16ELi64EEELi0ESW_Li16EEELSS_0ENS6_11threadblock9MmaPolicyINS6_4warp11MmaTensorOpINS7_ILi64ELi64ELi32EEESC_SO_SC_SZ_fNSF_8RowMajorENS13_17MmaTensorOpPolicyINSQ_3MmaINS7_ILi16ELi8ELi16EEELi32ESC_S16_SC_NSF_11ColumnMajorEfS16_NSQ_13OpMultiplyAddEEENSA_ILi1ELi1EEEEELi1ELb0EbEENSA_ILi0ELi0EEES1G_Li1EEELi4EbEENS_8epilogue11threadblock8EpilogueIS8_S1F_Li1ENS1K_22PredicatedTileIteratorINS1K_26OutputTileOptimalThreadMapINS1K_15OutputTileShapeILi256ELi8ELi1ELi1ELi1EEENS1O_ILi1ELi8ELi1ELi1ELi8EEELi128ELi4ELi32EEEfEENS1J_4warp24FragmentIteratorTensorOpIS15_S19_fNS_5ArrayIfLi4ELb1EEES16_EENS1T_20TileIteratorTensorOpIS15_S19_fS16_EENS1K_18SharedLoadIteratorINS1R_18CompactedThreadMapEfLi16EEENS1J_6thread17LinearCombinationIfLi4EffLNS_15FloatRoundStyleE2EEENSA_ILi0ELi8EEEEENS11_30GemmIdentityThreadblockSwizzleILi4EEELNS1_8OperatorE2ENS1_17Conv2dProblemSizeEEEEEvNT_6ParamsEˇ ÄÄ*Ä2)8èÄ@@èÄ@HèÄ@Xb<gradient_tape/model_7/conv2d_173/Conv2D/Conv2DBackpropFilterh
Ö
√void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*Ä2{8éê?@Ñ¯HáÄbmodel_7/concatenate_57/concathuZUÖB
·
ùvoid convolve_common_engine_float_NHWC<__half, __half, 128, 5, 5, 3, 3, 3, true, false, false, false, false>(int, int, int, __half const*, __half const*, int, __half*, conv_kernel_common_params, unsigned long long, unsigned long, float, float, bool, __half const*, __half const*, bool)PÄ*2ÄÄ8è‡>@è‡>Hè‡>Xbmodel_7/conv2d_173/Conv2Dhu  HB
~
.ampere_fp16_s16816gemm_fp16_256x64_ldg8_f2f_tn◊ÄÄ ÄÄ*Ä2¿8éê>@éê>Héê>Xb%gradient_tape/model_7/dense_21/MatMulh
∫
htensorflow::functor::ReluGradHalfKernelVector(Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2ÄÄ8ç¯;@ç¯;Hç¯;b,gradient_tape/model_7/activation_79/ReluGradhu  »B
Ö
√void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*Ä2{8çÄ:@É»Hábmodel_7/concatenate_59/concathuZUÖB
∫
htensorflow::functor::ReluGradHalfKernelVector(Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2ÄÄ8ç9@ç9Hç9b,gradient_tape/model_7/activation_80/ReluGradhu  »B
∫
htensorflow::functor::ReluGradHalfKernelVector(Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2ÄÄ8éË9@éË9HéË9b,gradient_tape/model_7/activation_78/ReluGradhu  »B
Ö
√void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*Ä2{8ç–9@É¿HÜ»bmodel_7/concatenate_58/concathuZUÖB
°
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_tn‚ÄÄ ÄÄ*Ä2Ä8ãà1@ãà1Hãà1Xb;gradient_tape/model_7/conv2d_177/Conv2D/Conv2DBackpropInputh
ã
:ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_ntÍÄÄ*Ä2Ä	8ã–/@ã–/Hã–/b'gradient_tape/model_7/dense_21/MatMul_1hugUÖA
{
:ampere_fp16_s16816gemm_fp16_128x64_ldg8_f2f_stages_64x4_nnñÄÄ ÄÄ*Ä28ä®/@ä®/Hä®/Xbmodel_7/dense_21/MatMulh
¶	
Ávoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*Ä2{8âÿ(@âÿ(Hâÿ(bmodel_7/activation_80/ReluhuZUÖB
¶	
Ávoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*Ä2{8ä®(@ä®(Hä®(bmodel_7/activation_79/ReluhuZUÖB
¶	
Ávoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*Ä2{8â¯'@â¯'Hâ¯'bmodel_7/activation_78/ReluhuZUÖB
°
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_tn‚ÄÄ ÄÄ*Ä2
Ä8âÄ&@âÄ&HâÄ&Xb;gradient_tape/model_7/conv2d_189/Conv2D/Conv2DBackpropInputh

;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_nnﬁÄÄ ÄÄ*Ä2Ä8â¯%@â¯%Hâ¯%Xbmodel_7/conv2d_189/Conv2Dh
ê
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2Ä¿8â‡$@É†HÉ†bAgradient_tape/model_7/batch_normalization_74/FusedBatchNormGradV3hu  »B
ê
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2Ä¿8âÿ$@ÉòHÉ®bAgradient_tape/model_7/batch_normalization_75/FusedBatchNormGradV3hu  »B
°
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_64x3_ntÊÄÄ ÄÄ*Ä2
8àË"@àË"HàË"Xb<gradient_tape/model_7/conv2d_189/Conv2D/Conv2DBackpropFilterh
ê
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2Ä†8à†!@ÉÄHÇòbAgradient_tape/model_7/batch_normalization_77/FusedBatchNormGradV3hu  »B
ö
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8àÄ!@àÄ!HàÄ!b,gradient_tape/model_7/concatenate_57/Slice_2huZUÖB

;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_nnﬁÄÄ ÄÄ*Ä2Ä8à¯ @à¯ Hà¯ Xbmodel_7/conv2d_177/Conv2Dh
ö
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8à¯ @à¯ Hà¯ b,gradient_tape/model_7/concatenate_59/Slice_1huZUÖB
ö
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8àÿ @àÿ Hàÿ b,gradient_tape/model_7/concatenate_58/Slice_1huZUÖB
°
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_64x3_ntÊÄÄ ÄÄ*Ä28á» @á» Há» Xb<gradient_tape/model_7/conv2d_177/Conv2D/Conv2DBackpropFilterh
Ì
Évoid cudnn::bn_bw_1C11_kernel_new<__half, float, float2, 128, true, 1>(float, float, float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, float const*, float*, float*, float const*, float const*, float)(ê*Ä2Ä8á»@á»Há»bAgradient_tape/model_7/batch_normalization_74/FusedBatchNormGradV3hu  »B
Ì
Évoid cudnn::bn_bw_1C11_kernel_new<__half, float, float2, 128, true, 1>(float, float, float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, float const*, float*, float*, float const*, float const*, float)(ê*Ä2Ä8á¿@á¿Há¿bAgradient_tape/model_7/batch_normalization_75/FusedBatchNormGradV3hu  »B
™
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Üò@ÜòHÜòbmodel_7/conv2d_175/BiasAddhuZUÖB
™
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8áà@áàHáàbmodel_7/conv2d_178/BiasAddhuZUÖB
™
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8áË@áËHáËbmodel_7/conv2d_173/BiasAddhuZUÖB
”
åvoid pooling_fw_kernel_max_nhwc<__half, float, 0, (cudnnNanPropagation_t)0, false>(PoolingParams, __half*, __half const*, float, float, int)*Ä2ÄÄ8Ü–@Ü–HÜ–b model_7/max_pooling2d_29/MaxPoolhu  »B
”
åvoid pooling_fw_kernel_max_nhwc<__half, float, 0, (cudnnNanPropagation_t)0, false>(PoolingParams, __half*, __half const*, float, float, int)*Ä2ÄÄ8ÜÄ@ÜÄHÜÄb model_7/max_pooling2d_28/MaxPoolhu  »B
ñ
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8ÜË@ÜËHÜËb<cond_1/then/_10/cond_1/Adam/Adam/update_50/ResourceApplyAdamhuZUÖB
˛
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2Ä¿8Ü–@É®HÉ®b/model_7/batch_normalization_75/FusedBatchNormV3hu  »B
˛
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2Ä¿8Ö»@É†HÇ®b/model_7/batch_normalization_74/FusedBatchNormV3hu  »B
≠
Àvoid cutlass_cudnn::Kernel<cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4> >(cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4>::Params)6* 2@?8ÖÄ@ÖÄHÖÄXb<gradient_tape/model_7/conv2d_185/Conv2D/Conv2DBackpropFilterhu  ñB
Ÿ

ü
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ö‡@Ö‡HÖ‡bAdam/gradients/AddN_6huZUÖB
˛
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2Ä†8Öò@ÇàHÉêb/model_7/batch_normalization_77/FusedBatchNormV3hu  »B
≠
Àvoid cutlass_cudnn::Kernel<cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4> >(cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4>::Params)6* 2@?8Ñÿ@ÑÿHÑÿXb<gradient_tape/model_7/conv2d_188/Conv2D/Conv2DBackpropFilterhu  ñB
å
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2ú8Öê@ÖêHÖêbAll_56hu  »B
ü
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8ÖÄ@ÖÄHÖÄb4gradient_tape/model_7/conv2d_173/BiasAdd/BiasAddGradhuZUÖB
ü
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Ñê@ÑêHÑêb4gradient_tape/model_7/conv2d_172/BiasAdd/BiasAddGradhuZUÖB
ü
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Öà@ÖàHÖàb4gradient_tape/model_7/conv2d_171/BiasAdd/BiasAddGradhuZUÖB
ü
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8ÑË@ÑËHÑËb4gradient_tape/model_7/conv2d_175/BiasAdd/BiasAddGradhuZUÖB
ü
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8ÑË@ÑËHÑËb4gradient_tape/model_7/conv2d_178/BiasAdd/BiasAddGradhuZUÖB
≠
Àvoid cutlass_cudnn::Kernel<cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4> >(cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4>::Params)6* 2 28Ñ¿@Ñ¿HÑ¿Xb<gradient_tape/model_7/conv2d_179/Conv2D/Conv2DBackpropFilterhu  ñB
≠
Àvoid cutlass_cudnn::Kernel<cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4> >(cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4>::Params)6* 2@28Öò@ÖòHÖòXb<gradient_tape/model_7/conv2d_182/Conv2D/Conv2DBackpropFilterhu  ñB
∞
ÿvoid cudnn::bn_fw_tr_1C11_kernel_NCHW<__half, float, 128, true, 1>(cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(ê*Ä2Ä8Ñê@ÑêHÑêb/model_7/batch_normalization_75/FusedBatchNormV3hu  »B
Ÿ

ü
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ñà@ÑàHÑàbAdam/gradients/AddN_7huZUÖB
∞
ÿvoid cudnn::bn_fw_tr_1C11_kernel_NCHW<__half, float, 128, true, 1>(cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, float const*, float const*, float, float, float*, float*, float*, float*, float, float)(ê*Ä2Ä8Ö¯@Ö¯HÖ¯b/model_7/batch_normalization_74/FusedBatchNormV3hu  »B
Ÿ

ü
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ö@ÖHÖbAdam/gradients/AddN_9huZUÖB
Æ
Nvoid cudnn::ops::scalePackedTensor_kernel<__half, float>(long, __half*, float)*Ä2ˇˇ8Ñ»@Ñ»HÑ»b:gradient_tape/model_7/max_pooling2d_28/MaxPool/MaxPoolGradhu  »B
ê
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2Ä†8É»@ÅòHÅòbAgradient_tape/model_7/batch_normalization_76/FusedBatchNormGradV3hu  »B
Æ
Nvoid cudnn::ops::scalePackedTensor_kernel<__half, float>(long, __half*, float)*Ä2ˇˇ8Ñ¿@Ñ¿HÑ¿b:gradient_tape/model_7/max_pooling2d_29/MaxPool/MaxPoolGradhu  »B
Ö
√void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*Ä2{8Ñ∏@Å‡HÅ∞bmodel_7/concatenate_60/concathuZUÖB
π
htensorflow::functor::ReluGradHalfKernelVector(Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2ÄP8Ö†@Ö†HÖ†b,gradient_tape/model_7/activation_81/ReluGradhu  »B
Ö
√void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*Ä2{8Ñ†@ÅËHÇ†bmodel_7/concatenate_61/concathuZUÖB
π
htensorflow::functor::ReluGradHalfKernelVector(Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2ÄP8Ñà@ÑàHÑàb,gradient_tape/model_7/activation_82/ReluGradhu  »B
ñ
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8Ñ∏@Ñ∏HÑ∏b<cond_1/then/_10/cond_1/Adam/Adam/update_52/ResourceApplyAdamhuZUÖB
Ä
ìvoid cudnn::bn_bw_1C11_singleread_specialized<__half2, 512, 1, 2, 14>(float, float, float, float, cudnnTensorStruct, __half2 const*, cudnnTensorStruct, __half2 const*, cudnnTensorStruct, __half2*, float const*, float*, float*, float const*, float const*, float, cudnn::reduced_divisor, int, cudnn::reduced_divisor, cudnn::bnBwPersistentState*, int, float, float, float, int, float, cudnnStatus_t*, bool)@ê Ä@*Ä2Ä8Ñ¯@Ñ¯HÑ¯bAgradient_tape/model_7/batch_normalization_77/FusedBatchNormGradV3huZUÖB
·
ùvoid convolve_common_engine_float_NHWC<__half, __half, 128, 5, 5, 3, 3, 3, true, false, false, false, false>(int, int, int, __half const*, __half const*, int, __half*, conv_kernel_common_params, unsigned long long, unsigned long, float, float, bool, __half const*, __half const*, bool)PÄ*2ÄÄ8É¯@É¯HÉ¯Xbmodel_7/conv2d_172/Conv2Dhu  HB
ò
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8É»@É»HÉ»b*gradient_tape/model_7/concatenate_57/SlicehuZUÖB
ö
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Ñ∏@Ñ∏HÑ∏b,gradient_tape/model_7/concatenate_57/Slice_1huZUÖB
π
htensorflow::functor::ReluGradHalfKernelVector(Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2ÄH8É∞@É∞HÉ∞b,gradient_tape/model_7/activation_84/ReluGradhu  »B
ò
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8É∞@É∞HÉ∞b*gradient_tape/model_7/concatenate_59/SlicehuZUÖB
Ö
√void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*Ä2{8Ñ®@Å¯HÇòbmodel_7/concatenate_63/concathuZUÖB
ò
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8É®@É®HÉ®b*gradient_tape/model_7/concatenate_58/SlicehuZUÖB
ö
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8É®@É®HÉ®b,gradient_tape/model_7/concatenate_59/Slice_2huZUÖB
ö
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8É†@É†HÉ†b,gradient_tape/model_7/concatenate_58/Slice_2huZUÖB
°
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_tn‚ÄÄ ÄÄ*Ä2Ä8Éà@ÉàHÉàXb;gradient_tape/model_7/conv2d_183/Conv2D/Conv2DBackpropInputh
c
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä@8É∏@É∏HÉ∏b model_7/dropout_43/dropout/Mul_1huZUÖB
q
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä@8Éÿ@ÉÿHÉÿb.gradient_tape/model_7/dropout_43/dropout/Mul_1huZUÖB
c
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä@8Éÿ@ÉÿHÉÿb model_7/dropout_42/dropout/Mul_1huZUÖB
q
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä@8É–@É–HÉ–b.gradient_tape/model_7/dropout_42/dropout/Mul_1huZUÖB
™
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Ñ¿@Ñ¿HÑ¿bmodel_7/conv2d_176/BiasAddhuZUÖB
™
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Ñ∞@Ñ∞HÑ∞bmodel_7/conv2d_174/BiasAddhuZUÖB
™
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8É∞@É∞HÉ∞bmodel_7/conv2d_177/BiasAddhuZUÖB
™
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8É∞@É∞HÉ∞bmodel_7/conv2d_179/BiasAddhuZUÖB
™
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Éê@ÉêHÉêbmodel_7/conv2d_171/BiasAddhuZUÖB
™
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8ÉÄ@ÉÄHÉÄbmodel_7/conv2d_172/BiasAddhuZUÖB
ø
dvoid tensorflow::BiasGradNHWC_SharedAtomics<Eigen::half>(int, Eigen::half const*, Eigen::half*, int) Ä*Ä2R8ÉË@ÉËHÉËb4gradient_tape/model_7/conv2d_187/BiasAdd/BiasAddGradhuZUÖB
°
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_ntﬁÄÄ ÄÄ*Ä28É∞@É∞HÉ∞Xb<gradient_tape/model_7/conv2d_186/Conv2D/Conv2DBackpropFilterh
ø
dvoid tensorflow::BiasGradNHWC_SharedAtomics<Eigen::half>(int, Eigen::half const*, Eigen::half*, int) Ä*Ä2R8É®@É®HÉ®b4gradient_tape/model_7/conv2d_186/BiasAdd/BiasAddGradhuZUÖB
°
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_tn‚ÄÄ ÄÄ*Ä2Ä8É¯@É¯HÉ¯Xb;gradient_tape/model_7/conv2d_180/Conv2D/Conv2DBackpropInputh
≠
Àvoid cutlass_cudnn::Kernel<cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4> >(cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4>::Params)6* 2@}8É@ÉHÉXb<gradient_tape/model_7/conv2d_191/Conv2D/Conv2DBackpropFilterhu  ñB
±
Ìvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8É‡@É‡HÉ‡bmodel_7/dropout_43/dropout/CasthuZUÖB
¶	
Ávoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*Ä2{8É–@É–HÉ–bmodel_7/activation_81/ReluhuZUÖB
¶	
Ávoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*Ä2{8É–@É–HÉ–bmodel_7/activation_82/ReluhuZUÖB
±
Ìvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ç»@Ç»HÇ»bmodel_7/dropout_42/dropout/CasthuZUÖB
˛
•void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<Eigen::half, 256, 32, 32, false>(Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*) ¿*Ä2Ä†8Ñ¿@ÇòHÇ®b/model_7/batch_normalization_76/FusedBatchNormV3hu  »B
Ÿ

ü
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8ÉÄ@ÉÄHÉÄbAdam/gradients/AddN_4huZUÖB
°
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_64x3_ntÊÄÄ ÄÄ*Ä28É¯@É¯HÉ¯Xb<gradient_tape/model_7/conv2d_183/Conv2D/Conv2DBackpropFilterh
°
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_tn‚ÄÄ ÄÄ*Ä2Ä8Ç¯@Ç¯HÇ¯Xb;gradient_tape/model_7/conv2d_186/Conv2D/Conv2DBackpropInputh
–
ıvoid cudnn::bn_fw_tr_1C11_singleread_specialized<__half2, 512, 1, 2, 10>(cudnnTensorStruct, __half2 const*, cudnnTensorStruct, __half2*, float const*, float const*, float, float, float*, float*, float*, float*, float, float, cudnn::reduced_divisor, int, cudnn::reduced_divisor, cudnn::bnFwPersistentState*, int, float, float, float, int, float, float, cudnnStatus_t*, bool)>ê Ä`*Ä2Ä8ÉË@ÉËHÉËb/model_7/batch_normalization_77/FusedBatchNormV3huZUÖB
¶	
Ávoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ç¿@Ç¿HÇ¿bmodel_7/activation_84/ReluhuZUÖB

;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_nnﬁÄÄ ÄÄ*Ä2Ä8É®@É®HÉ®Xbmodel_7/conv2d_183/Conv2Dh

;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_nnﬁÄÄ ÄÄ*Ä2Ä8Éê@ÉêHÉêXbmodel_7/conv2d_186/Conv2Dh
ó
1ampere_s16816gemm_fp16_128x64_ldg8_stages_64x4_ntíÄÄ ÄÄ*Ä28Éê@ÉêHÉêXb<gradient_tape/model_7/conv2d_174/Conv2D/Conv2DBackpropFilterh
™
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8É∏
@É∏
HÉ∏
bAdam/gradients/AddN_29huZUÖB
ü
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8É∏
@É∏
HÉ∏
b4gradient_tape/model_7/conv2d_176/BiasAdd/BiasAddGradhuZUÖB
ü
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8É∏
@É∏
HÉ∏
b4gradient_tape/model_7/conv2d_177/BiasAdd/BiasAddGradhuZUÖB
ü
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Ç∏
@Ç∏
HÇ∏
b4gradient_tape/model_7/conv2d_179/BiasAdd/BiasAddGradhuZUÖB
˙
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ç∏
@Ç∏
HÇ∏
Xb<gradient_tape/model_7/conv2d_190/Conv2D/Conv2DBackpropFilterhuZUÖB
v
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2¿8É®
@É®
HÉ®
b1gradient_tape/conv2d_190/kernel/Regularizer/Mul_1huZUÖB
ü
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Ç®
@Ç®
HÇ®
b4gradient_tape/model_7/conv2d_174/BiasAdd/BiasAddGradhuZUÖB
s
.ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_nnˆÄÄ*Ä2Ä8Çê
@Çê
HÇê
Xbmodel_7/conv2d_174/Conv2DhugUÖA
ï
.ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_tn˛ÄÄ*Ä2Ä8Çê
@Çê
HÇê
Xb;gradient_tape/model_7/conv2d_174/Conv2D/Conv2DBackpropInputhugUÖA
ü
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Çê
@Çê
HÇê
b4gradient_tape/model_7/conv2d_189/BiasAdd/BiasAddGradhuZUÖB
ü
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Çà
@Çà
HÇà
b4gradient_tape/model_7/conv2d_190/BiasAdd/BiasAddGradhuZUÖB
Ä
ìvoid cudnn::bn_bw_1C11_singleread_specialized<__half2, 512, 1, 2, 14>(float, float, float, float, cudnnTensorStruct, __half2 const*, cudnnTensorStruct, __half2 const*, cudnnTensorStruct, __half2*, float const*, float*, float*, float const*, float const*, float, cudnn::reduced_divisor, int, cudnn::reduced_divisor, cudnn::bnBwPersistentState*, int, float, float, float, int, float, cudnnStatus_t*, bool)@ê Ä@*Ä2Ä
8ÉÄ
@ÉÄ
HÉÄ
bAgradient_tape/model_7/batch_normalization_76/FusedBatchNormGradV3huZUÖB
o
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä@8Ç¯	@Ç¯	HÇ¯	b,gradient_tape/model_7/dropout_42/dropout/MulhuZUÖB
o
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä@8Ç¯	@Ç¯	HÇ¯	b,gradient_tape/model_7/dropout_43/dropout/MulhuZUÖB
a
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä@8É	@É	HÉ	bmodel_7/dropout_42/dropout/MulhuZUÖB
a
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä@8Ç	@Ç	HÇ	bmodel_7/dropout_43/dropout/MulhuZUÖB
◊
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ç‡	@Ç‡	HÇ‡	Xbmodel_7/conv2d_190/Conv2DhuZUÖB
˘
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Çÿ	@Çÿ	HÇÿ	Xb;gradient_tape/model_7/conv2d_190/Conv2D/Conv2DBackpropInputhuZUÖB

;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_nnﬁÄÄ ÄÄ*Ä2Ä8É†	@É†	HÉ†	Xbmodel_7/conv2d_180/Conv2Dh
π
htensorflow::functor::ReluGradHalfKernelVector(Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2Ä(8Ç†	@Ç†	HÇ†	b,gradient_tape/model_7/activation_83/ReluGradhu  »B
Ö
√void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*Ä2{8Çò	@ÄHÅÿbmodel_7/concatenate_62/concathuZUÖB
ñ
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8Ç¯@Ç¯HÇ¯b<cond_1/then/_10/cond_1/Adam/Adam/update_60/ResourceApplyAdamhuZUÖB
ñ
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8Ç@ÇHÇb<cond_1/then/_10/cond_1/Adam/Adam/update_36/ResourceApplyAdamhuZUÖB
≠
Àvoid cutlass_cudnn::Kernel<cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4> >(cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4>::Params)6* 2@8Ç‡@Ç‡HÇ‡Xb<gradient_tape/model_7/conv2d_184/Conv2D/Conv2DBackpropFilterhu  ñB
ñ
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8Ç‡@Ç‡HÇ‡b<cond_1/then/_10/cond_1/Adam/Adam/update_44/ResourceApplyAdamhuZUÖB
°
;ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_64x3_ntÊÄÄ ÄÄ*Ä28Çÿ@ÇÿHÇÿXb<gradient_tape/model_7/conv2d_180/Conv2D/Conv2DBackpropFilterh
”
åvoid pooling_fw_kernel_max_nhwc<__half, float, 0, (cudnnNanPropagation_t)0, false>(PoolingParams, __half*, __half const*, float, float, int)*Ä2Ä¿8Ç¿@Ç¿HÇ¿b model_7/max_pooling2d_30/MaxPoolhu  »B
ÿ
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2{8Å¿@Å¿HÅ¿b0gradient_tape/conv2d_190/kernel/Regularizer/TilehuZUÖB
ö
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Ç∞@Ç∞HÇ∞b,gradient_tape/model_7/concatenate_60/Slice_2huZUÖB
ö
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Å∞@Å∞HÅ∞b,gradient_tape/model_7/concatenate_60/Slice_1huZUÖB
ö
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Å∞@Å∞HÅ∞b,gradient_tape/model_7/concatenate_61/Slice_2huZUÖB
ö
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Å®@Å®HÅ®b,gradient_tape/model_7/concatenate_61/Slice_1huZUÖB
É
ßvoid tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>)*Ä2{8Çê@ÇêHÇêb7model_7/dropout_43/dropout/random_uniform/RandomUniformhuZUÖB
ò
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8ÇÄ@ÇÄHÇÄb*gradient_tape/model_7/concatenate_63/SlicehuZUÖB
ö
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8ÅÄ@ÅÄHÅÄb,gradient_tape/model_7/concatenate_63/Slice_1huZUÖB
É
ßvoid tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>)*Ä2{8ÇË@ÇËHÇËb7model_7/dropout_42/dropout/random_uniform/RandomUniformhuZUÖB
”
åvoid pooling_fw_kernel_max_nhwc<__half, float, 0, (cudnnNanPropagation_t)0, false>(PoolingParams, __half*, __half const*, float, float, int)*Ä2Ä†8Ç‡@Ç‡HÇ‡b model_7/max_pooling2d_31/MaxPoolhu  »B
s
'GreaterEqual_GPU_DT_HALF_DT_BOOL_kernel*Ä2Ä@8Ç∞@Ç∞HÇ∞b'model_7/dropout_43/dropout/GreaterEqualhuZUÖB
™
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ç®@Ç®HÇ®bAdam/gradients/AddN_30huZUÖB
™
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Ç®@Ç®HÇ®bmodel_7/conv2d_182/BiasAddhuZUÖB
™
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Ç®@Ç®HÇ®bmodel_7/conv2d_190/BiasAddhuZUÖB
s
'GreaterEqual_GPU_DT_HALF_DT_BOOL_kernel*Ä2Ä@8Å®@Å®HÅ®b'model_7/dropout_42/dropout/GreaterEqualhuZUÖB
™
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Ç†@Ç†HÇ†bmodel_7/conv2d_189/BiasAddhuZUÖB
™
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Å†@Å†HÅ†bmodel_7/conv2d_181/BiasAddhuZUÖB
™
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Å†@Å†HÅ†bmodel_7/conv2d_185/BiasAddhuZUÖB
v
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2–8Çò@ÇòHÇòb1gradient_tape/conv2d_191/kernel/Regularizer/Mul_1huZUÖB
™
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Çò@ÇòHÇòbmodel_7/conv2d_184/BiasAddhuZUÖB
≠
Àvoid cutlass_cudnn::Kernel<cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4> >(cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4>::Params)6* 2@8Çê@ÇêHÇêXb<gradient_tape/model_7/conv2d_181/Conv2D/Conv2DBackpropFilterhu  ñB
ñ
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8Åê@ÅêHÅêb<cond_1/then/_10/cond_1/Adam/Adam/update_28/ResourceApplyAdamhuZUÖB
t
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2¿8ÇÄ@ÇÄHÇÄb/gradient_tape/conv2d_190/kernel/Regularizer/MulhuZUÖB
l
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2¿8ÇÄ@ÇÄHÇÄb$conv2d_190/kernel/Regularizer/SquarehuZUÖB
K
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2¿8ÅÄ@ÅÄHÅÄbmul_52huZUÖB
d
 ampere_fp16_sgemm_fp16_32x128_nn9ÄÄ*Ä2Ä@8Å@ÅHÅXbmodel_7/conv2d_171/Conv2DhuZUÖB
–
ıvoid cudnn::bn_fw_tr_1C11_singleread_specialized<__half2, 512, 1, 2, 10>(cudnnTensorStruct, __half2 const*, cudnnTensorStruct, __half2*, float const*, float const*, float, float, float*, float*, float*, float*, float, float, cudnn::reduced_divisor, int, cudnn::reduced_divisor, cudnn::bnFwPersistentState*, int, float, float, float, int, float, float, cudnnStatus_t*, bool)>ê Ä`*Ä2Ä
8Å‡@Å‡HÅ‡b/model_7/batch_normalization_76/FusedBatchNormV3huZUÖB
◊
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Å‡@Å‡HÅ‡Xbmodel_7/conv2d_191/Conv2DhuZUÖB
˘
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Çÿ@ÇÿHÇÿXb;gradient_tape/model_7/conv2d_191/Conv2D/Conv2DBackpropInputhuZUÖB
π
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ç¿@Ç¿HÇ¿b1gradient_tape/model_7/conv2d_190/Conv2D/Cast/CasthuZUÖB
ñ
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8Ç∏@Ç∏HÇ∏b<cond_1/then/_10/cond_1/Adam/Adam/update_42/ResourceApplyAdamhuZUÖB
¶	
Ávoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å®@Å®HÅ®bmodel_7/activation_83/ReluhuZUÖB
Ÿ

ü
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half, Eigen::half>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<Eigen::half const, Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Åê@ÅêHÅêbAdam/gradients/AddN_5huZUÖB
≤
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å¯@Å¯HÅ¯bmodel_7/conv2d_190/Conv2D/CasthuZUÖB
ÿ
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2{8Ç@ÇHÇb0gradient_tape/conv2d_191/kernel/Regularizer/TilehuZUÖB
Æ
Nvoid cudnn::ops::scalePackedTensor_kernel<__half, float>(long, __half*, float)*Ä2ˇˇ8Ç@ÇHÇb:gradient_tape/model_7/max_pooling2d_30/MaxPool/MaxPoolGradhu  »B
ç
§void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nt_align1>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nt_align1::Params)¢ Ä¿*Ä2¢8ÅË@ÅËHÅËXb<gradient_tape/model_7/conv2d_171/Conv2D/Conv2DBackpropFilterhugUÖA
ü
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Ç∏@Ç∏HÇ∏b4gradient_tape/model_7/conv2d_184/BiasAdd/BiasAddGradhuZUÖB
ü
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Ç∞@Ç∞HÇ∞b4gradient_tape/model_7/conv2d_181/BiasAdd/BiasAddGradhuZUÖB
Æ
Nvoid cudnn::ops::scalePackedTensor_kernel<__half, float>(long, __half*, float)*Ä2ˇˇ8Å®@Å®HÅ®b:gradient_tape/model_7/max_pooling2d_31/MaxPool/MaxPoolGradhu  »B
ü
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Å®@Å®HÅ®b4gradient_tape/model_7/conv2d_182/BiasAdd/BiasAddGradhuZUÖB
ü
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Å®@Å®HÅ®b4gradient_tape/model_7/conv2d_185/BiasAdd/BiasAddGradhuZUÖB
Ì
âvoid cudnn::ops::convertTensor_kernel<float, __half, float, (cudnnKernelDataType_t)0>(float, float const*, float, __half*, unsigned long)*Ä2Ä 8Çò@ÇòHÇòXb<gradient_tape/model_7/conv2d_190/Conv2D/Conv2DBackpropFilterhu  »B
˙
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Åà@ÅàHÅàXb<gradient_tape/model_7/conv2d_191/Conv2D/Conv2DBackpropFilterhuZUÖB
K
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2–8Å¯@Å¯HÅ¯bmul_54huZUÖB
t
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2–8Å@ÅHÅb/gradient_tape/conv2d_191/kernel/Regularizer/MulhuZUÖB
l
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2–8Å@ÅHÅb$conv2d_191/kernel/Regularizer/SquarehuZUÖB
q
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä8Å‡@Å‡HÅ‡b.gradient_tape/model_7/dropout_44/dropout/Mul_1huZUÖB
c
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä8Åÿ@ÅÿHÅÿb model_7/dropout_44/dropout/Mul_1huZUÖB
π
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å¿@Å¿HÅ¿b1gradient_tape/model_7/conv2d_191/Conv2D/Cast/CasthuZUÖB
ò
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Å∏@Å∏HÅ∏b*gradient_tape/model_7/concatenate_62/SlicehuZUÖB
ö
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Å∏@Å∏HÅ∏b,gradient_tape/model_7/concatenate_62/Slice_1huZUÖB
T
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2ÄZ8Å®@Å®HÅ®bIsFinite_50hu  »B
c
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä8Å†@Å†HÅ†b model_7/dropout_45/dropout/Mul_1huZUÖB
ò
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Å†@Å†HÅ†b*gradient_tape/model_7/concatenate_60/SlicehuZUÖB
ò
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Å†@Å†HÅ†b*gradient_tape/model_7/concatenate_61/SlicehuZUÖB
q
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä8Ä†@Ä†HÄ†b.gradient_tape/model_7/dropout_45/dropout/Mul_1huZUÖB
≤
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Åò@ÅòHÅòbmodel_7/conv2d_191/Conv2D/CasthuZUÖB
±
Ìvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Åê@ÅêHÅêbmodel_7/dropout_44/dropout/CasthuZUÖB
™
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å@ÅHÅbAdam/gradients/AddN_32huZUÖB
˘
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Å@ÅHÅXb;gradient_tape/model_7/conv2d_188/Conv2D/Conv2DBackpropInputhuZUÖB
t
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ä8ÅË@ÅËHÅËb/gradient_tape/dense_22/kernel/Regularizer/Mul_1huZUÖB
™
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8ÅË@ÅËHÅËbAdam/gradients/AddN_27huZUÖB
™
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8ÅË@ÅËHÅËbmodel_7/conv2d_187/BiasAddhuZUÖB
v
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ë8Å‡@Å‡HÅ‡b1gradient_tape/conv2d_188/kernel/Regularizer/Mul_1huZUÖB
™
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Å‡@Å‡HÅ‡bmodel_7/conv2d_180/BiasAddhuZUÖB
™
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Å‡@Å‡HÅ‡bmodel_7/conv2d_186/BiasAddhuZUÖB
™
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä‡@Ä‡HÄ‡bAdam/gradients/AddN_24huZUÖB
±
Ìvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Åÿ@ÅÿHÅÿbmodel_7/dropout_45/dropout/CasthuZUÖB
v
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ë8Äÿ@ÄÿHÄÿb1gradient_tape/conv2d_185/kernel/Regularizer/Mul_1huZUÖB
™
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Äÿ@ÄÿHÄÿbmodel_7/conv2d_183/BiasAddhuZUÖB
ñ
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8Å–@Å–HÅ–b<cond_1/then/_10/cond_1/Adam/Adam/update_20/ResourceApplyAdamhuZUÖB
˘
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ä»@Ä»HÄ»Xb;gradient_tape/model_7/conv2d_185/Conv2D/Conv2DBackpropInputhuZUÖB
◊
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Å∏@Å∏HÅ∏Xbmodel_7/conv2d_188/Conv2DhuZUÖB
ƒ
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2ú8Å∞@Å∞HÅ∞b!conv2d_190/kernel/Regularizer/Sumhu  »B
◊
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Å∞@Å∞HÅ∞Xbmodel_7/conv2d_185/Conv2DhuZUÖB
ñ
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8Å†@Å†HÅ†b<cond_1/then/_10/cond_1/Adam/Adam/update_34/ResourceApplyAdamhuZUÖB
a
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä8Åò@ÅòHÅòbmodel_7/dropout_44/dropout/MulhuZUÖB
o
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä8Äò@ÄòHÄòb,gradient_tape/model_7/dropout_44/dropout/MulhuZUÖB
T
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2¿>8Åà@ÅàHÅàbIsFinite_52hu  »B
™
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Åà@ÅàHÅàbAdam/gradients/AddN_21huZUÖB
ÿ
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2{8ÅÄ@ÅÄHÅÄb0gradient_tape/conv2d_185/kernel/Regularizer/TilehuZUÖB
ñ
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8ÅÄ@ÅÄHÅÄb<cond_1/then/_10/cond_1/Adam/Adam/update_48/ResourceApplyAdamhuZUÖB
v
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8ÄÄ@ÄÄHÄÄb1gradient_tape/conv2d_182/kernel/Regularizer/Mul_1huZUÖB
o
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä8Ä¯@Ä¯HÄ¯b,gradient_tape/model_7/dropout_45/dropout/MulhuZUÖB
a
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2Ä8Å@ÅHÅbmodel_7/dropout_45/dropout/MulhuZUÖB
ü
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Å@ÅHÅb4gradient_tape/model_7/conv2d_183/BiasAdd/BiasAddGradhuZUÖB
˘
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ä@ÄHÄXb;gradient_tape/model_7/conv2d_182/Conv2D/Conv2DBackpropInputhuZUÖB
ü
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8ÅË@ÅËHÅËb4gradient_tape/model_7/conv2d_180/BiasAdd/BiasAddGradhuZUÖB
˘
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ä‡@Ä‡HÄ‡Xb;gradient_tape/model_7/conv2d_187/Conv2D/Conv2DBackpropInputhuZUÖB
v
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2–8Åÿ@ÅÿHÅÿb1gradient_tape/conv2d_187/kernel/Regularizer/Mul_1huZUÖB
™
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Åÿ@ÅÿHÅÿbAdam/gradients/AddN_26huZUÖB
ñ
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8Åÿ@ÅÿHÅÿb<cond_1/then/_10/cond_1/Adam/Adam/update_26/ResourceApplyAdamhuZUÖB
◊
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Äÿ@ÄÿHÄÿXbmodel_7/conv2d_182/Conv2DhuZUÖB
t
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ë8Å»@Å»HÅ»b/gradient_tape/conv2d_188/kernel/Regularizer/MulhuZUÖB
K
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ë8Å»@Å»HÅ»bmul_46huZUÖB
r
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ä8Å»@Å»HÅ»b-gradient_tape/dense_22/kernel/Regularizer/MulhuZUÖB
K
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ä8Å»@Å»HÅ»bmul_62huZUÖB
˙
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Å»@Å»HÅ»Xb<gradient_tape/model_7/conv2d_188/Conv2D/Conv2DBackpropFilterhuZUÖB
j
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ä8Ä»@Ä»HÄ»b"dense_22/kernel/Regularizer/SquarehuZUÖB
t
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ë8Å¿@Å¿HÅ¿b/gradient_tape/conv2d_185/kernel/Regularizer/MulhuZUÖB
ÿ
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2{8Å¿@Å¿HÅ¿b0gradient_tape/conv2d_182/kernel/Regularizer/TilehuZUÖB
ñ
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8Å¿@Å¿HÅ¿b<cond_1/then/_10/cond_1/Adam/Adam/update_18/ResourceApplyAdamhuZUÖB
É
ßvoid tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>)*Ä2{8Å¿@Å¿HÅ¿b7model_7/dropout_44/dropout/random_uniform/RandomUniformhuZUÖB
˙
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Å¿@Å¿HÅ¿Xb<gradient_tape/model_7/conv2d_185/Conv2D/Conv2DBackpropFilterhuZUÖB
K
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ë8Ä¿@Ä¿HÄ¿bmul_38huZUÖB
l
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ë8Ä¿@Ä¿HÄ¿b$conv2d_188/kernel/Regularizer/SquarehuZUÖB
◊
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ä¿@Ä¿HÄ¿Xbmodel_7/conv2d_187/Conv2DhuZUÖB
ƒ
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2–8Å∏@Å∏HÅ∏b!conv2d_191/kernel/Regularizer/Sumhu  »B
l
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ë8Ä∏@Ä∏HÄ∏b$conv2d_185/kernel/Regularizer/SquarehuZUÖB
›
˙void nhwcAddPaddingKernel<__half, __half, float, true, (cudnnKernelDataType_t)0>(int, int, int, int, int, int, int, int, __half const*, __half*, int, int, int, int, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor)*Ä2§8Å∞@ÄHÅòXb<gradient_tape/model_7/conv2d_173/Conv2D/Conv2DBackpropFilterhu  »B
É
ßvoid tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>)*Ä2{8Å∞@Å∞HÅ∞b7model_7/dropout_45/dropout/random_uniform/RandomUniformhuZUÖB
s
'GreaterEqual_GPU_DT_HALF_DT_BOOL_kernel*Ä2Ä8Ä∞@Ä∞HÄ∞b'model_7/dropout_44/dropout/GreaterEqualhuZUÖB
π
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å®@Å®HÅ®b1gradient_tape/model_7/conv2d_185/Conv2D/Cast/CasthuZUÖB
π
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å®@Å®HÅ®b1gradient_tape/model_7/conv2d_188/Conv2D/Cast/CasthuZUÖB
ö
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Ä®@Ä®HÄ®b,gradient_tape/model_7/concatenate_62/Slice_2huZUÖB
∑
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä®@Ä®HÄ®b/gradient_tape/model_7/dense_22/MatMul/Cast/CasthuZUÖB
ÿ
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2{8Ä®@Ä®HÄ®b0gradient_tape/conv2d_187/kernel/Regularizer/TilehuZUÖB
≤
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å†@Å†HÅ†bmodel_7/conv2d_185/Conv2D/CasthuZUÖB
∞
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å†@Å†HÅ†bmodel_7/dense_22/MatMul/CasthuZUÖB
≤
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Åò@ÅòHÅòbmodel_7/conv2d_188/Conv2D/CasthuZUÖB
s
'GreaterEqual_GPU_DT_HALF_DT_BOOL_kernel*Ä2Ä8Äò@ÄòHÄòb'model_7/dropout_45/dropout/GreaterEqualhuZUÖB
K
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8Åê@ÅêHÅêbmul_30huZUÖB
ö
…void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Äê@ÄêHÄêb,gradient_tape/model_7/concatenate_63/Slice_2huZUÖB
˙
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Åà@ÅàHÅàXb<gradient_tape/model_7/conv2d_187/Conv2D/Conv2DBackpropFilterhuZUÖB
t
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8Äà@ÄàHÄàb/gradient_tape/conv2d_182/kernel/Regularizer/MulhuZUÖB
l
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8ÅÄ@ÅÄHÅÄb$conv2d_182/kernel/Regularizer/SquarehuZUÖB
™
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8ÄÄ@ÄÄHÄÄbmodel_7/conv2d_188/BiasAddhuZUÖB
˙
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8ÄÄ@ÄÄHÄÄXb<gradient_tape/model_7/conv2d_182/Conv2D/Conv2DBackpropFilterhuZUÖB
™
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Ä¯@Ä¯HÄ¯bmodel_7/conv2d_191/BiasAddhuZUÖB
t
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2–8Å@ÅHÅb/gradient_tape/conv2d_187/kernel/Regularizer/MulhuZUÖB
K
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2–8Å@ÅHÅbmul_44huZUÖB
π
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å@ÅHÅb1gradient_tape/model_7/conv2d_182/Conv2D/Cast/CasthuZUÖB
≤
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8ÄË@ÄËHÄËbmodel_7/conv2d_182/Conv2D/CasthuZUÖB
l
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2–8Ä‡@Ä‡HÄ‡b$conv2d_187/kernel/Regularizer/SquarehuZUÖB
´
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä‡@Ä‡HÄ‡bmodel_7/conv2d_171/CasthuZUÖB
π
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Åÿ@ÅÿHÅÿb1gradient_tape/model_7/conv2d_187/Conv2D/Cast/CasthuZUÖB
ò
Ávoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<unsigned char const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<unsigned char const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Åÿ@ÅÿHÅÿbmodel_7/CasthuZUÖB
T
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2Ä 8Äÿ@ÄÿHÄÿbIsFinite_60hu  »B
ü
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Äÿ@ÄÿHÄÿb4gradient_tape/model_7/conv2d_191/BiasAdd/BiasAddGradhuZUÖB
T
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2†8Å–@Å–HÅ–bIsFinite_36hu  »B
≤
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å–@Å–HÅ–bmodel_7/conv2d_187/Conv2D/CasthuZUÖB
™
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å–@Å–HÅ–bAdam/gradients/AddN_18huZUÖB
ü
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2 8Å–@Å–HÅ–b4gradient_tape/model_7/conv2d_188/BiasAdd/BiasAddGradhuZUÖB
T
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2†8Ä–@Ä–HÄ–bIsFinite_44hu  »B
v
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2ê8Å»@Å»HÅ»b1gradient_tape/conv2d_179/kernel/Regularizer/Mul_1huZUÖB
÷
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Ä»@Ä»HÄ»b.gradient_tape/dense_22/kernel/Regularizer/TilehuZUÖB
˘
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Å¿@Å¿HÅ¿Xb;gradient_tape/model_7/conv2d_179/Conv2D/Conv2DBackpropInputhuZUÖB
™
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä¿@Ä¿HÄ¿bAdam/gradients/AddN_23huZUÖB
Ñ
Campere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x5_nníÄÄ ÄÄ*Ä2 8Ä∏@Ä∏HÄ∏Xbmodel_7/dense_22/MatMulh
í
Campere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tníÄÄ ÄÄ*Ä2 8Ä∏@Ä∏HÄ∏Xb%gradient_tape/model_7/dense_22/MatMulh
ƒ
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2Ë8Ä∏@Ä∏HÄ∏b!conv2d_188/kernel/Regularizer/Sumhu  »B
˘
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ä∏@Ä∏HÄ∏Xb;gradient_tape/model_7/conv2d_184/Conv2D/Conv2DBackpropInputhuZUÖB
◊
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ä∏@Ä∏HÄ∏Xbmodel_7/conv2d_179/Conv2DhuZUÖB
v
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ë8Ä∞@Ä∞HÄ∞b1gradient_tape/conv2d_184/kernel/Regularizer/Mul_1huZUÖB
ƒ
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2Ë8Ä∞@Ä∞HÄ∞b!conv2d_185/kernel/Regularizer/Sumhu  »B
T
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2Ä8Ä®@Ä®HÄ®bIsFinite_28hu  »B
™
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä®@Ä®HÄ®bAdam/gradients/AddN_28huZUÖB
ÿ
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2{8Ä®@Ä®HÄ®b0gradient_tape/conv2d_179/kernel/Regularizer/TilehuZUÖB
¬
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2Ä8Ä®@Ä®HÄ®bdense_22/kernel/Regularizer/Sumhu  »B
◊
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ä®@Ä®HÄ®Xbmodel_7/conv2d_184/Conv2DhuZUÖB
™
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å†@Å†HÅ†bAdam/gradients/AddN_20huZUÖB
T
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2¿8Ä†@Ä†HÄ†bIsFinite_42hu  »B
v
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2¿8Ä†@Ä†HÄ†b1gradient_tape/conv2d_189/kernel/Regularizer/Mul_1huZUÖB
ÿ
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2{8Ä†@Ä†HÄ†b0gradient_tape/conv2d_184/kernel/Regularizer/TilehuZUÖB
ƒ
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2†8Ä†@Ä†HÄ†b!conv2d_182/kernel/Regularizer/Sumhu  »B
™
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Åò@ÅòHÅòbAdam/gradients/AddN_17huZUÖB
å
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2†8Åò@ÅòHÅòbAll_50hu  »B
˙
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Åò@ÅòHÅòXb<gradient_tape/model_7/conv2d_179/Conv2D/Conv2DBackpropFilterhuZUÖB
˘
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Äò@ÄòHÄòXb;gradient_tape/model_7/conv2d_178/Conv2D/Conv2DBackpropInputhuZUÖB
˘
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Äò@ÄòHÄòXb;gradient_tape/model_7/conv2d_181/Conv2D/Conv2DBackpropInputhuZUÖB
◊
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Äò@ÄòHÄòXbmodel_7/conv2d_181/Conv2DhuZUÖB
v
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8Åê@ÅêHÅêb1gradient_tape/conv2d_178/kernel/Regularizer/Mul_1huZUÖB
v
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8Äê@ÄêHÄêb1gradient_tape/conv2d_181/kernel/Regularizer/Mul_1huZUÖB
t
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2ê8Äê@ÄêHÄêb/gradient_tape/conv2d_179/kernel/Regularizer/MulhuZUÖB
ÿ
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2{8Äê@ÄêHÄêb0gradient_tape/conv2d_189/kernel/Regularizer/TilehuZUÖB
ñ
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8Äê@ÄêHÄêb<cond_1/then/_10/cond_1/Adam/Adam/update_12/ResourceApplyAdamhuZUÖB
◊
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Äê@ÄêHÄêXbmodel_7/conv2d_178/Conv2DhuZUÖB
K
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2ê8Äà@ÄàHÄàbmul_22huZUÖB
l
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2ê8Äà@ÄàHÄàb$conv2d_179/kernel/Regularizer/SquarehuZUÖB
ÿ
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2{8Äà@ÄàHÄàb0gradient_tape/conv2d_178/kernel/Regularizer/TilehuZUÖB
≤
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8ÅÄ@ÅÄHÅÄbmodel_7/conv2d_179/Conv2D/CasthuZUÖB
ƒ
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2–8ÅÄ@ÅÄHÅÄb!conv2d_187/kernel/Regularizer/Sumhu  »B
t
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ë8ÄÄ@ÄÄHÄÄb/gradient_tape/conv2d_184/kernel/Regularizer/MulhuZUÖB
K
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ë8ÄÄ@ÄÄHÄÄbmul_36huZUÖB
l
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ë8ÄÄ@ÄÄHÄÄb$conv2d_184/kernel/Regularizer/SquarehuZUÖB
ä
:ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_ntÍÄÄ*Ä28ÄÄ@ÄÄHÄÄb'gradient_tape/model_7/dense_22/MatMul_1hugUÖA
q
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2¿8Äx@ÄxHÄxb/gradient_tape/conv2d_189/kernel/Regularizer/MulhuZUÖB
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Äx@ÄxHÄxbmodel_7/conv2d_184/Conv2D/CasthuZUÖB
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Äx@ÄxHÄxb1gradient_tape/model_7/conv2d_179/Conv2D/Cast/CasthuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8Äx@ÄxHÄxb<cond_1/then/_10/cond_1/Adam/Adam/update_40/ResourceApplyAdamhuZUÖB
˜
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Äx@ÄxHÄxXb<gradient_tape/model_7/conv2d_184/Conv2D/Conv2DBackpropFilterhuZUÖB
Ì
ëvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorForcedEvalOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorForcedEvalOp<Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_log_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorForcedEvalOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorForcedEvalOp<Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_log_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long)2*Ä28Åp@ÅpHÅpb:categorical_crossentropy/softmax_cross_entropy_with_logitshuZUÖB
q
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8Äp@ÄpHÄpb/gradient_tape/conv2d_178/kernel/Regularizer/MulhuZUÖB
H
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2¿8Äp@ÄpHÄpbmul_50huZUÖB
Ã
Êvoid xmma_cudnn::gemm::split_k_kernel<xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, xmma_cudnn::Col, 256, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 3> >(xmma_cudnn::implicit_gemm::wgrad_indexed::Kernel_traits<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_a<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, 16, xmma_cudnn::Col, 256, 32> >, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, false, xmma_cudnn::implicit_gemm::wgrad_indexed::Gmem_tile_base_b<xmma_cudnn::Ampere_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Ampere, 256, 128, 32, 4, 2, 1, 1>, false, 16, xmma_cudnn::Row, 128, 32> >, false, 3>::Params)? Äê*Ä2	8Äp@ÄpHÄpPXb<gradient_tape/model_7/conv2d_175/Conv2D/Conv2DBackpropFilterhuMUB
q
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8Äh@ÄhHÄhb/gradient_tape/conv2d_181/kernel/Regularizer/MulhuZUÖB
H
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8Äh@ÄhHÄhbmul_20huZUÖB
H
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8Äh@ÄhHÄhbmul_28huZUÖB
i
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2¿8Äh@ÄhHÄhb$conv2d_189/kernel/Regularizer/SquarehuZUÖB
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Äh@ÄhHÄhbmodel_7/conv2d_189/Conv2D/CasthuZUÖB
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Äh@ÄhHÄhb1gradient_tape/model_7/conv2d_184/Conv2D/Cast/CasthuZUÖB
â
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2Ë8Äh@ÄhHÄhbAll_52hu  »B
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2R8Äh@ÄhHÄhb<cond_1/then/_10/cond_1/Adam/Adam/update_10/ResourceApplyAdamhuZUÖB
˜
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Äh@ÄhHÄhXb<gradient_tape/model_7/conv2d_178/Conv2D/Conv2DBackpropFilterhuZUÖB
˜
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Äh@ÄhHÄhXb<gradient_tape/model_7/conv2d_181/Conv2D/Conv2DBackpropFilterhuZUÖB
Q
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2Ä
8Ä`@Ä`HÄ`bIsFinite_48hu  »B
i
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8Ä`@Ä`HÄ`b$conv2d_178/kernel/Regularizer/SquarehuZUÖB
i
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2†8Ä`@Ä`HÄ`b$conv2d_181/kernel/Regularizer/SquarehuZUÖB
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä`@Ä`HÄ`bmodel_7/conv2d_178/Conv2D/CasthuZUÖB
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä`@Ä`HÄ`bmodel_7/conv2d_181/Conv2D/CasthuZUÖB
Q
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2¿8ÅX@ÅXHÅXbIsFinite_20hu  »B
Q
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2†8ÄX@ÄXHÄXbIsFinite_34hu  »B
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8ÄX@ÄXHÄXb1gradient_tape/model_7/conv2d_189/Conv2D/Cast/CasthuZUÖB
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8ÅP@ÅPHÅPb1gradient_tape/model_7/conv2d_178/Conv2D/Cast/CasthuZUÖB
’
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2{8ÅP@ÅPHÅPb0gradient_tape/conv2d_176/kernel/Regularizer/TilehuZUÖB
Q
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2Ä	8ÄP@ÄPHÄPbIsFinite_26hu  »B
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8ÄP@ÄPHÄPb1gradient_tape/model_7/conv2d_181/Conv2D/Cast/CasthuZUÖB
ß
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8ÅH@ÅHHÅHbAdam/gradients/AddN_15huZUÖB
ˆ
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8ÅH@ÅHHÅHXb;gradient_tape/model_7/conv2d_176/Conv2D/Conv2DBackpropInputhuZUÖB
Q
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2Ä	8ÄH@ÄHHÄHbIsFinite_18hu  »B
â
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2Ù8ÄH@ÄHHÄHbAll_44hu  »B
¡
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2†8ÄH@ÄHHÄHb!conv2d_178/kernel/Regularizer/Sumhu  »B
¡
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2¿8ÄH@ÄHHÄHb!conv2d_189/kernel/Regularizer/Sumhu  »B
¡
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2Ë8ÄH@ÄHHÄHb!conv2d_184/kernel/Regularizer/Sumhu  »B
¡
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2ê8ÄH@ÄHHÄHb!conv2d_179/kernel/Regularizer/Sumhu  »B
â
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2ê8Å@@Å@HÅ@bAll_28hu  »B
¡
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2†8Å@@Å@HÅ@b!conv2d_181/kernel/Regularizer/Sumhu  »B
‘
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Å@@Å@HÅ@Xbmodel_7/conv2d_176/Conv2DhuZUÖB
r
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2d8Ä@@Ä@HÄ@b1gradient_tape/conv2d_176/kernel/Regularizer/Mul_1huZUÖB
ß
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä@@Ä@HÄ@bAdam/gradients/AddN_25huZUÖB
Î
èvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorReshapingOp<Eigen::IndexList<int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> >, Eigen::TensorReductionOp<Eigen::internal::MaxReducer<float, 0>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorReshapingOp<Eigen::IndexList<int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> >, Eigen::TensorReductionOp<Eigen::internal::MaxReducer<float, 0>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long)@*Ä28Ä@@Ä@HÄ@b:categorical_crossentropy/softmax_cross_entropy_with_logitshuZUÖB
â
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2Ù8Ä@@Ä@HÄ@bAll_36hu  »B
ˆ
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Å8@Å8HÅ8Xb;gradient_tape/model_7/conv2d_175/Conv2D/Conv2DBackpropInputhuZUÖB
Â
âvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorReshapingOp<Eigen::IndexList<int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> >, Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorReshapingOp<Eigen::IndexList<int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> >, Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long)/*Ä28Ä8@Ä8HÄ8b:categorical_crossentropy/softmax_cross_entropy_with_logitshuZUÖB
â
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2Ë8Ä8@Ä8HÄ8bAll_42hu  »B
‘
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ä8@Ä8HÄ8Xbmodel_7/conv2d_175/Conv2DhuZUÖB
ä
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Å0@Å0HÅ0b4gradient_tape/model_7/conv2d_182/BiasAdd/BiasAddGradhuZUÖB
r
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2H8Ä0@Ä0HÄ0b1gradient_tape/conv2d_175/kernel/Regularizer/Mul_1huZUÖB
r
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2P8Ä0@Ä0HÄ0b1gradient_tape/conv2d_186/kernel/Regularizer/Mul_1huZUÖB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2d8Ä0@Ä0HÄ0b/gradient_tape/conv2d_176/kernel/Regularizer/MulhuZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2d8Ä0@Ä0HÄ0bmul_14huZUÖB
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä0@Ä0HÄ0b1gradient_tape/model_7/conv2d_176/Conv2D/Cast/CasthuZUÖB
ß
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä0@Ä0HÄ0bAdam/gradients/AddN_14huZUÖB
’
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä28Ä0@Ä0HÄ0b0gradient_tape/conv2d_174/kernel/Regularizer/TilehuZUÖB
’
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2@8Ä0@Ä0HÄ0b0gradient_tape/conv2d_177/kernel/Regularizer/TilehuZUÖB
’
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2P8Ä0@Ä0HÄ0b0gradient_tape/conv2d_183/kernel/Regularizer/TilehuZUÖB
’
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2{8Ä0@Ä0HÄ0b0gradient_tape/conv2d_175/kernel/Regularizer/TilehuZUÖB
’
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä2{8Ä0@Ä0HÄ0b0gradient_tape/conv2d_186/kernel/Regularizer/TilehuZUÖB
â
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2†8Ä0@Ä0HÄ0bAll_48hu  »B
â
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2Ä8Ä0@Ä0HÄ0bAll_60hu  »B
Í
âvoid cudnn::ops::convertTensor_kernel<float, __half, float, (cudnnKernelDataType_t)0>(float, float const*, float, __half*, unsigned long)*Ä2Ä 8Ä0@Ä0HÄ0Xb<gradient_tape/model_7/conv2d_172/Conv2D/Conv2DBackpropFilterhu  »B
™
Àvoid cutlass_cudnn::Kernel<cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4> >(cutlass_cudnn::reduction::kernel::ReduceSplitK<cutlass_cudnn::MatrixShape<4, 256>, cutlass_cudnn::epilogue::thread::LinearCombination<cutlass_cudnn::half_t, 8, float, float, (cutlass_cudnn::FloatRoundStyle)2>, cutlass_cudnn::reduction::thread::ReduceAdd<float, float, 8>, 4>::Params)6* 28Ä0@Ä0HÄ0Xb<gradient_tape/model_7/conv2d_173/Conv2D/Conv2DBackpropFilterhu  ñB
Ñ
§void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*) *Ä28Ä0@Ä0HÄ0Xb<gradient_tape/model_7/conv2d_171/Conv2D/Conv2DBackpropFilterhu  »B
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2@8Ä0@Ä0HÄ0b<cond_1/then/_10/cond_1/Adam/Adam/update_24/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2P8Ä0@Ä0HÄ0b<cond_1/then/_10/cond_1/Adam/Adam/update_32/ResourceApplyAdamhuZUÖB
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä2 8Ä0@Ä0HÄ0bAll_32hu  »B
ä
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Ä0@Ä0HÄ0b4gradient_tape/model_7/conv2d_181/BiasAdd/BiasAddGradhuZUÖB
ä
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Ä0@Ä0HÄ0b4gradient_tape/model_7/conv2d_185/BiasAdd/BiasAddGradhuZUÖB
˜
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ä0@Ä0HÄ0Xb<gradient_tape/model_7/conv2d_176/Conv2D/Conv2DBackpropFilterhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Å(@Å(HÅ(b<cond_1/then/_10/cond_1/Adam/Adam/update_57/ResourceApplyAdamhuZUÖB
Q
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2†8Ä(@Ä(HÄ(bIsFinite_10hu  »B
Q
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2¿8Ä(@Ä(HÄ(bIsFinite_40hu  »B
Q
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2ê8Ä(@Ä(HÄ(bIsFinite_12hu  »B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2H8Ä(@Ä(HÄ(b/gradient_tape/conv2d_175/kernel/Regularizer/MulhuZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2H8Ä(@Ä(HÄ(bmul_12huZUÖB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2P8Ä(@Ä(HÄ(b/gradient_tape/conv2d_186/kernel/Regularizer/MulhuZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2P8Ä(@Ä(HÄ(bmul_42huZUÖB
h
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2P8Ä(@Ä(HÄ(b$conv2d_186/kernel/Regularizer/SquarehuZUÖB
h
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2d8Ä(@Ä(HÄ(b$conv2d_176/kernel/Regularizer/SquarehuZUÖB
›
ìvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long) *Ä28Ä(@ÄHÄb(ArithmeticOptimizer/AddOpsRewrite_AddN_1huZUÖB
’
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä28Ä(@Ä(HÄ(b0gradient_tape/conv2d_173/kernel/Regularizer/TilehuZUÖB
â
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2ê8Ä(@Ä(HÄ(bAll_18hu  »B
â
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2ê8Ä(@Ä(HÄ(bAll_26hu  »B
â
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2¥8Ä(@Ä(HÄ(bAll_34hu  »B
â
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2»8Ä(@Ä(HÄ(bAll_20hu  »B
¿
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2d8Ä(@Ä(HÄ(b!conv2d_176/kernel/Regularizer/Sumhu  »B
„
§void cutlass::Kernel<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2>(cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2::Params)_ Ä§*Ä28Ä(@Ä(HÄ(Xbmodel_7/dense_23/MatMulhugUÖA
Ö
§void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*) *Ä2Ä8Ä(@Ä(HÄ(Xb<gradient_tape/model_7/conv2d_177/Conv2D/Conv2DBackpropFilterhu  »B
Ö
§void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*) *Ä2Ä8Ä(@Ä(HÄ(Xb<gradient_tape/model_7/conv2d_186/Conv2D/Conv2DBackpropFilterhu  »B
Ë
©void tensorflow::(anonymous namespace)::GenerateNormalizedProb<Eigen::half, float, 8>(Eigen::half const*, float const*, Eigen::half const*, Eigen::half*, int, int, bool)*Ä28Ä(@Ä(HÄ(bmodel_7/activation_87/Softmaxhu  »B
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä(@Ä(HÄ(b<cond_1/then/_10/cond_1/Adam/Adam/update_49/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä(@Ä(HÄ(b<cond_1/then/_10/cond_1/Adam/Adam/update_53/ResourceApplyAdamhuZUÖB
í
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä(@Ä(HÄ(b;cond_1/then/_10/cond_1/Adam/Adam/update_6/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä2@8Ä(@Ä(HÄ(b<cond_1/then/_10/cond_1/Adam/Adam/update_16/ResourceApplyAdamhuZUÖB
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä(@Ä(HÄ(bAll_54hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä(@Ä(HÄ(bAll_55hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä(@Ä(HÄ(bAll_57hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä(@Ä(HÄ(bAll_61hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä2 8Ä(@Ä(HÄ(bAll_16hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä2 8Ä(@Ä(HÄ(bAll_24hu  »B
ä
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Ä(@Ä(HÄ(b4gradient_tape/model_7/conv2d_175/BiasAdd/BiasAddGradhuZUÖB
ä
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Ä(@Ä(HÄ(b4gradient_tape/model_7/conv2d_178/BiasAdd/BiasAddGradhuZUÖB
ä
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Ä(@Ä(HÄ(b4gradient_tape/model_7/conv2d_184/BiasAdd/BiasAddGradhuZUÖB
ä
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Ä(@Ä(HÄ(b4gradient_tape/model_7/conv2d_188/BiasAdd/BiasAddGradhuZUÖB
ä
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Ä(@Ä(HÄ(b4gradient_tape/model_7/conv2d_191/BiasAdd/BiasAddGradhuZUÖB
˜
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä2R8Ä(@Ä(HÄ(Xb<gradient_tape/model_7/conv2d_175/Conv2D/Conv2DBackpropFilterhuZUÖB
r
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Å @Å HÅ b1gradient_tape/conv2d_180/kernel/Regularizer/Mul_1huZUÖB
í
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Å @Å HÅ b;cond_1/then/_10/cond_1/Adam/Adam/update_2/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Å @Å HÅ b<cond_1/then/_10/cond_1/Adam/Adam/update_29/ResourceApplyAdamhuZUÖB
©
√void tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)Ä!*  2@8Å @Å HÅ bBgradient_tape/model_7/batch_normalization_78/batchnorm/add_1/Sum_1huZUÖB
r
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ä8Ä @Ä HÄ b.model_7/batch_normalization_78/batchnorm/add_1huZUÖB
r
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ä8Ä @Ä HÄ b.model_7/batch_normalization_79/batchnorm/add_1huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä @Ä HÄ bmul_18huZUÖB
r
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä @Ä HÄ b1gradient_tape/conv2d_183/kernel/Regularizer/Mul_1huZUÖB
Ç
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ä8Ä @Ä HÄ b@gradient_tape/model_7/batch_normalization_78/batchnorm/mul_1/MulhuZUÖB
Ç
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ä8Ä @Ä HÄ b@gradient_tape/model_7/batch_normalization_79/batchnorm/mul_1/MulhuZUÖB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ä8Ä @Ä HÄ b.model_7/batch_normalization_79/batchnorm/mul_1huZUÖB
k
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2 8Ä @Ä HÄ b,gradient_tape/model_7/dropout_46/dropout/MulhuZUÖB
h
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2H8Ä @Ä HÄ b$conv2d_175/kernel/Regularizer/SquarehuZUÖB
z
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ä8Ä @Ä HÄ b8gradient_tape/model_7/batch_normalization_79/moments/subhuZUÖB
l
 ampere_fp16_sgemm_fp16_128x32_tn9ÄÄ*Ä28Ä @Ä HÄ Xb%gradient_tape/model_7/dense_23/MatMulhuZUÖB
`
 ampere_fp16_sgemm_fp16_32x128_nn9ÄÄ*Ä28Ä @Ä HÄ Xbmodel_7/conv2d_171/Conv2DhuZUÖB
Æ
Ìvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä @Ä HÄ bmodel_7/dropout_47/dropout/CasthuZUÖB
Ω
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä @Ä HÄ b8gradient_tape/model_7/batch_normalization_78/Cast_1/CasthuZUÖB
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä @Ä HÄ b1gradient_tape/model_7/conv2d_175/Conv2D/Cast/CasthuZUÖB
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä @Ä HÄ b1gradient_tape/model_7/conv2d_186/Conv2D/Cast/CasthuZUÖB
ß
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä @Ä HÄ bAdam/gradients/AddN_13huZUÖB
ß
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä @Ä HÄ bAdam/gradients/AddN_16huZUÖB
ß
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä @Ä HÄ bAdam/gradients/AddN_19huZUÖB
ß
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2P8Ä @Ä HÄ bAdam/gradients/AddN_22huZUÖB
ˆ	
ø	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä @Ä HÄ bAdam/gradients/AddN_3huZUÖB
”
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä28Ä @Ä HÄ b.gradient_tape/dense_23/kernel/Regularizer/TilehuZUÖB
Á
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Ä @Ä HÄ bBgradient_tape/model_7/batch_normalization_78/moments/BroadcastTo_1huZUÖB
Â
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Ä @Ä HÄ b@gradient_tape/model_7/batch_normalization_79/moments/BroadcastTohuZUÖB
¡	
Ávoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_squared_difference_op<float>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_squared_difference_op<float>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, int) *Ä2{8Ä @Ä HÄ b8model_7/batch_normalization_78/moments/SquaredDifferencehuZUÖB
≈
Èvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_quotient_op<float, float>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_quotient_op<float, float>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long).*Ä28Ä @Ä HÄ b:categorical_crossentropy/softmax_cross_entropy_with_logitshuZUÖB
’
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä28Ä @Ä HÄ b0gradient_tape/conv2d_171/kernel/Regularizer/TilehuZUÖB
≠
—void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorEvalToOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorForcedEvalOp<Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_log_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorEvalToOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorForcedEvalOp<Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_log_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>, long)(*Ä28Ä @Ä HÄ b:categorical_crossentropy/softmax_cross_entropy_with_logitshuZUÖB
à
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä228Ä @Ä HÄ bAll_12hu  »B
¿
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2H8Ä @Ä HÄ b!conv2d_175/kernel/Regularizer/Sumhu  »B
¿
˚void cub::DeviceReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float> >(float*, float*, int, cub::GridEvenShare<int>, tensorflow::functor::Sum<float>)0*Ä2P8Ä @Ä HÄ b!conv2d_186/kernel/Regularizer/Sumhu  »B
Ö
€void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *Ä28Ä @Ä HÄ bAll_56hu¶™¶B
Ò
§void cutlass::Kernel<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nt_align2>(cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nt_align2::Params)_ Ä¿*Ä2@8Ä @Ä HÄ b'gradient_tape/model_7/dense_23/MatMul_1hugUÖA
Ö
§void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*) *Ä2Ä8Ä @Ä HÄ Xb<gradient_tape/model_7/conv2d_180/Conv2D/Conv2DBackpropFilterhu  »B
Ö
§void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*) *Ä2Ä8Ä @Ä HÄ Xb<gradient_tape/model_7/conv2d_183/Conv2D/Conv2DBackpropFilterhu  »B
‡
§void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*) *Ä2Ä8Ä @Ä HÄ Xbmodel_7/dense_21/MatMulhu  »B
É
¢void splitKreduce_kernel<float, __half, float, __half>(cublasSplitKParams<float>, float const*, __half const*, __half*, float const*, float const*, __half const*) *Ä2Ä8Ä @Ä HÄ Xb<gradient_tape/model_7/conv2d_174/Conv2D/Conv2DBackpropFilterhu  »B
ê
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b9cond_1/then/_10/cond_1/Adam/Adam/update/ResourceApplyAdamhuZUÖB
í
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b;cond_1/then/_10/cond_1/Adam/Adam/update_1/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_13/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_15/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_19/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_23/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_25/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_27/ResourceApplyAdamhuZUÖB
í
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b;cond_1/then/_10/cond_1/Adam/Adam/update_3/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_30/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_35/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_37/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_38/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_41/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_43/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_45/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_51/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_65/ResourceApplyAdamhuZUÖB
í
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b;cond_1/then/_10/cond_1/Adam/Adam/update_7/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä @Ä HÄ b<cond_1/then/_10/cond_1/Adam/Adam/update_58/ResourceApplyAdamhuZUÖB
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
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä @Ä HÄ bAll_43hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä @Ä HÄ bAll_46hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä @Ä HÄ bAll_47hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä @Ä HÄ bAll_49hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä @Ä HÄ bAll_51hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä @Ä HÄ bAll_59hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä @Ä HÄ bAll_62hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä @Ä HÄ bAll_63hu  »B
á
¬void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*Ä2 8Ä @Ä HÄ b!conv2d_180/kernel/Regularizer/Sumhu  »B
ä
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Ä @Ä HÄ b4gradient_tape/model_7/conv2d_174/BiasAdd/BiasAddGradhuZUÖB
ä
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Ä @Ä HÄ b4gradient_tape/model_7/conv2d_177/BiasAdd/BiasAddGradhuZUÖB
ä
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Ä @Ä HÄ b4gradient_tape/model_7/conv2d_179/BiasAdd/BiasAddGradhuZUÖB
ä
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Ä @Ä HÄ b4gradient_tape/model_7/conv2d_183/BiasAdd/BiasAddGradhuZUÖB
–
Åvoid tensorflow::functor::ColumnReduceKernel<float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, tensorflow::functor::Sum<float> >(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)Ä!*  2@8Ä @Ä HÄ b+model_7/batch_normalization_79/moments/meanhuZUÖB
Ä
ßvoid tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>)*Ä2{8Ä @Ä HÄ b7model_7/dropout_46/dropout/random_uniform/RandomUniformhuZUÖB
Ä
ßvoid tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, Eigen::half>)*Ä2{8Ä @Ä HÄ b7model_7/dropout_47/dropout/random_uniform/RandomUniformhuZUÖB
‘
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä28Ä @Ä HÄ Xbmodel_7/conv2d_172/Conv2DhuZUÖB
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Å@ÅHÅbIsFinite_51hu  »B
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Å@ÅHÅbmul_2huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Å@ÅHÅbmul_41huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Å@ÅHÅbmul_59huZUÖB
£	
Ávoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å@ÅHÅbmodel_7/activation_85/ReluhuZUÖB
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2P8Å@ÅHÅb1gradient_tape/model_7/conv2d_183/Conv2D/Cast/CasthuZUÖB
Ö
€void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *Ä28Å@ÅHÅbAll_40hu¶™¶B
ª
évoid tensorflow::(anonymous namespace)::concat_fixed_kernel<bool, int>(tensorflow::GpuDeviceArrayStruct<bool const*, 8>, int, int, int, bool*)*B28Å@ÅHÅbAll_66/inputhu∞öB
Ö
¬void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*Ä2 8Å@ÅHÅbdense_23/kernel/Regularizer/Sumhu  »B
ö
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2@8Å@ÅHÅb2gradient_tape/model_7/dense_22/BiasAdd/BiasAddGradhuZUÖB
‘
Åvoid tensorflow::functor::ColumnReduceKernel<float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, tensorflow::functor::Sum<float> >(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)Ä!*  2@8Å@ÅHÅb/model_7/batch_normalization_79/moments/variancehuZUÖB
˜
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä28Å@ÅHÅXb<gradient_tape/model_7/conv2d_172/Conv2D/Conv2DBackpropFilterhuZUÖB
o
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb,model_7/batch_normalization_79/batchnorm/addhuZUÖB
X
"AddV2_GPU_DT_INT64_DT_INT64_kernel*Ä28Ä@ÄHÄbcond/then/_0/cond/addhuZUÖB
~
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ä8Ä@ÄHÄb<gradient_tape/model_7/batch_normalization_78/moments/truedivhuZUÖB
Ä
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ä8Ä@ÄHÄb>gradient_tape/model_7/batch_normalization_78/moments/truediv_1huZUÖB
~
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ä8Ä@ÄHÄb<gradient_tape/model_7/batch_normalization_79/moments/truedivhuZUÖB
Ä
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ä8Ä@ÄHÄb>gradient_tape/model_7/batch_normalization_79/moments/truediv_1huZUÖB
o
'GreaterEqual_GPU_DT_HALF_DT_BOOL_kernel*Ä2 8Ä@ÄHÄb'model_7/dropout_46/dropout/GreaterEqualhuZUÖB
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_11hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_15hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_21hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_27hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_29hu  »B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄb
IsFinite_3hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_31hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_35hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_37hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_41hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_43hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_45hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_49hu  »B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄb
IsFinite_5hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_65hu  »B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄb
IsFinite_7hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_46hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_47hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_57hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_63hu  »B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄb
IsFinite_4hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2@8Ä@ÄHÄbIsFinite_16hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2@8Ä@ÄHÄbIsFinite_24hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä2P8Ä@ÄHÄbIsFinite_32hu  »B
ç
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbLgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mulhuZUÖB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb!conv2d_174/kernel/Regularizer/mulhuZUÖB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb!conv2d_175/kernel/Regularizer/mulhuZUÖB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb!conv2d_182/kernel/Regularizer/mulhuZUÖB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb!conv2d_183/kernel/Regularizer/mulhuZUÖB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb!conv2d_188/kernel/Regularizer/mulhuZUÖB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb!conv2d_191/kernel/Regularizer/mulhuZUÖB
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbdense_21/kernel/Regularizer/mulhuZUÖB
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbdense_23/kernel/Regularizer/mulhuZUÖB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb/gradient_tape/conv2d_172/kernel/Regularizer/MulhuZUÖB
Å
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb@gradient_tape/model_7/batch_normalization_78/batchnorm/mul/Mul_1huZUÖB
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb4model_7/batch_normalization_78/AssignMovingAvg_1/mulhuZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_11huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_13huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_15huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_16huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_17huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_19huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_21huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_24huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_32huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_35huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_39huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_40huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_45huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_47huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_48huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_49huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_53huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_55huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_56huZUÖB
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_8huZUÖB
r
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb1gradient_tape/conv2d_171/kernel/Regularizer/Mul_1huZUÖB

 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb>gradient_tape/model_7/batch_normalization_79/batchnorm/mul/MulhuZUÖB
m
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb,model_7/batch_normalization_78/batchnorm/mulhuZUÖB
m
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb,model_7/batch_normalization_79/batchnorm/mulhuZUÖB
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb.model_7/batch_normalization_79/batchnorm/mul_2huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_64huZUÖB
r
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb1gradient_tape/conv2d_173/kernel/Regularizer/Mul_1huZUÖB
n
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb-gradient_tape/dense_23/kernel/Regularizer/MulhuZUÖB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb/gradient_tape/dense_23/kernel/Regularizer/Mul_1huZUÖB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb/gradient_tape/conv2d_174/kernel/Regularizer/MulhuZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_10huZUÖB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb/gradient_tape/conv2d_177/kernel/Regularizer/MulhuZUÖB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb/gradient_tape/conv2d_180/kernel/Regularizer/MulhuZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_26huZUÖB
r
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb1gradient_tape/conv2d_177/kernel/Regularizer/Mul_1huZUÖB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb/gradient_tape/conv2d_183/kernel/Regularizer/MulhuZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_34huZUÖB
y
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2 8Ä@ÄHÄb8gradient_tape/model_7/batch_normalization_78/moments/MulhuZUÖB
y
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2 8Ä@ÄHÄb8gradient_tape/model_7/batch_normalization_79/moments/MulhuZUÖB
É
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2 8Ä@ÄHÄbBgradient_tape/model_7/batch_normalization_78/batchnorm/mul_1/Mul_1huZUÖB
{
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2 8Ä@ÄHÄb:gradient_tape/model_7/batch_normalization_78/moments/mul_1huZUÖB
É
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2 8Ä@ÄHÄbBgradient_tape/model_7/batch_normalization_79/batchnorm/mul_1/Mul_1huZUÖB
{
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2 8Ä@ÄHÄb:gradient_tape/model_7/batch_normalization_79/moments/mul_1huZUÖB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ä8Ä@ÄHÄb.model_7/batch_normalization_78/batchnorm/mul_1huZUÖB
m
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2 8Ä@ÄHÄb.gradient_tape/model_7/dropout_46/dropout/Mul_1huZUÖB
k
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2 8Ä@ÄHÄb,gradient_tape/model_7/dropout_47/dropout/MulhuZUÖB
m
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2 8Ä@ÄHÄb.gradient_tape/model_7/dropout_47/dropout/Mul_1huZUÖB
_
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2 8Ä@ÄHÄb model_7/dropout_46/dropout/Mul_1huZUÖB
_
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2 8Ä@ÄHÄb model_7/dropout_47/dropout/Mul_1huZUÖB
`
 Pow_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbcond_1/then/_10/cond_1/Adam/PowhuZUÖB
q
"Rsqrt_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb.model_7/batch_normalization_79/batchnorm/Rsqrthu  »B
h
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb$conv2d_172/kernel/Regularizer/SquarehuZUÖB
f
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb"dense_23/kernel/Regularizer/SquarehuZUÖB
h
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb$conv2d_174/kernel/Regularizer/SquarehuZUÖB
h
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb$conv2d_177/kernel/Regularizer/SquarehuZUÖB
h
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb$conv2d_180/kernel/Regularizer/SquarehuZUÖB
h
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb$conv2d_183/kernel/Regularizer/SquarehuZUÖB
s
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb2model_7/batch_normalization_78/AssignMovingAvg/subhuZUÖB
u
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb4model_7/batch_normalization_78/AssignMovingAvg_1/subhuZUÖB
m
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb,model_7/batch_normalization_78/batchnorm/subhuZUÖB
s
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb2model_7/batch_normalization_79/AssignMovingAvg/subhuZUÖB
z
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä2Ä8Ä@ÄHÄb8gradient_tape/model_7/batch_normalization_78/moments/subhuZUÖB
µ
htensorflow::functor::ReluGradHalfKernelVector(Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2 8Ä@ÄHÄb,gradient_tape/model_7/activation_85/ReluGradhu  »B
Æ
Ìvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä@ÄHÄbmodel_7/dropout_46/dropout/CasthuZUÖB
ï
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbCasthuZUÖB
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_7/conv2d_172/Conv2D/CasthuZUÖB
∞
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_7/conv2d_175/BiasAdd/CasthuZUÖB
∞
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_7/conv2d_181/BiasAdd/CasthuZUÖB
∞
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_7/conv2d_186/BiasAdd/CasthuZUÖB
∞
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_7/conv2d_191/BiasAdd/CasthuZUÖB
Æ
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_7/dense_22/BiasAdd/CasthuZUÖB
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄbmodel_7/conv2d_180/Conv2D/CasthuZUÖB
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2P8Ä@ÄHÄbmodel_7/conv2d_183/Conv2D/CasthuZUÖB
∂
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä@ÄHÄb%model_7/batch_normalization_78/Cast_1huZUÖB
∂
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä@ÄHÄb%model_7/batch_normalization_79/Cast_1huZUÖB
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä@ÄHÄbmodel_7/conv2d_175/Conv2D/CasthuZUÖB
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä@ÄHÄbmodel_7/conv2d_176/Conv2D/CasthuZUÖB
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä@ÄHÄbmodel_7/conv2d_186/Conv2D/CasthuZUÖB
£	
Ávoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<Eigen::half const, Eigen::half const, 1>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<Eigen::half const>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä@ÄHÄbmodel_7/activation_86/ReluhuZUÖB
ì
≈void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb,categorical_crossentropy/weighted_loss/valuehuZUÖB
Û
≈void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbdiv_no_nan_1huZUÖB
¨
≈void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbEgradient_tape/categorical_crossentropy/weighted_loss/value/div_no_nanhuZUÖB
Ω
€void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_rsqrt_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_rsqrt_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb@gradient_tape/model_7/batch_normalization_78/batchnorm/RsqrtGradhuZUÖB
ã
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbCast_6huZUÖB
Ç
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb}categorical_crossentropy/softmax_cross_entropy_with_logits/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Cast_1huZUÖB
∑
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb2gradient_tape/model_7/conv2d_171/BiasAdd/Cast/CasthuZUÖB
∑
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb2gradient_tape/model_7/conv2d_172/BiasAdd/Cast/CasthuZUÖB
∑
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb2gradient_tape/model_7/conv2d_175/BiasAdd/Cast/CasthuZUÖB
∑
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb2gradient_tape/model_7/conv2d_180/BiasAdd/Cast/CasthuZUÖB
∑
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb2gradient_tape/model_7/conv2d_182/BiasAdd/Cast/CasthuZUÖB
∑
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb2gradient_tape/model_7/conv2d_184/BiasAdd/Cast/CasthuZUÖB
∑
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb2gradient_tape/model_7/conv2d_185/BiasAdd/Cast/CasthuZUÖB
∑
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb2gradient_tape/model_7/conv2d_186/BiasAdd/Cast/CasthuZUÖB
∑
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb2gradient_tape/model_7/conv2d_187/BiasAdd/Cast/CasthuZUÖB
∑
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb2gradient_tape/model_7/conv2d_188/BiasAdd/Cast/CasthuZUÖB
∑
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb2gradient_tape/model_7/conv2d_189/BiasAdd/Cast/CasthuZUÖB
µ
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb0gradient_tape/model_7/dense_22/BiasAdd/Cast/CasthuZUÖB
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄb1gradient_tape/model_7/conv2d_177/Conv2D/Cast/CasthuZUÖB
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄb1gradient_tape/model_7/conv2d_180/Conv2D/Cast/CasthuZUÖB
Ω
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä@ÄHÄb8gradient_tape/model_7/batch_normalization_79/Cast_1/CasthuZUÖB
˚
”void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbCast_2huZUÖB
≠
”void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb8categorical_crossentropy/weighted_loss/num_elements/CasthuZUÖB
Æ
”void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb9gradient_tape/model_7/batch_normalization_78/moments/CasthuZUÖB
∞
”void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb;gradient_tape/model_7/batch_normalization_78/moments/Cast_1huZUÖB
∞
”void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb;gradient_tape/model_7/batch_normalization_79/moments/Cast_1huZUÖB
ô
’void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb"cond_1/then/_10/cond_1/Adam/Cast_1huZUÖB
√
Òvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb0model_7/batch_normalization_78/AssignMovingAvg_1huZUÖB
ß
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbAdam/gradients/AddN_11huZUÖB
ß
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbAdam/gradients/AddN_12huZUÖB
ß
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbAdam/gradients/AddN_33huZUÖB
ò
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbAssignAddVariableOphuZUÖB
ö
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbAssignAddVariableOp_3huZUÖB
ˆ	
ø	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä@ÄHÄbAdam/gradients/AddN_1huZUÖB
È
üvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long) *Ä28Ä@ÄHÄb(ArithmeticOptimizer/AddOpsRewrite_AddN_1huZUÖB
Â
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Ä@ÄHÄb@gradient_tape/model_7/batch_normalization_78/moments/BroadcastTohuZUÖB
Á
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä2{8Ä@ÄHÄbBgradient_tape/model_7/batch_normalization_79/moments/BroadcastTo_1huZUÖB
¡	
Ávoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_squared_difference_op<float>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_squared_difference_op<float>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, int) *Ä2{8Ä@ÄHÄb8model_7/batch_normalization_79/moments/SquaredDifferencehuZUÖB
ˇ	
£	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)&*Ä28Ä@ÄHÄb:categorical_crossentropy/softmax_cross_entropy_with_logitshuZUÖB
’
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *Ä28Ä@ÄHÄb0gradient_tape/conv2d_172/kernel/Regularizer/TilehuZUÖB
≥
ãvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorTupleReducerOp<Eigen::internal::ArgMaxTupleReducer<Eigen::Tuple<long, float> >, Eigen::array<long, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorTupleReducerOp<Eigen::internal::ArgMaxTupleReducer<Eigen::Tuple<long, float> >, Eigen::array<long, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long) *Ä28Ä@ÄHÄbArgMaxhuZUÖB
µ
ãvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorTupleReducerOp<Eigen::internal::ArgMaxTupleReducer<Eigen::Tuple<long, float> >, Eigen::array<long, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorTupleReducerOp<Eigen::internal::ArgMaxTupleReducer<Eigen::Tuple<long, float> >, Eigen::array<long, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long) *Ä28Ä@ÄHÄbArgMax_1huZUÖB
à
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2$8Ä@ÄHÄbAll_10hu  »B
à
ﬁvoid cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And) *Ä2(8Ä@ÄHÄbAll_40hu  »B
Ö
€void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *Ä28Ä@ÄHÄbAll_10hu¶™¶B
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
€void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)) *Ä28Ä@ÄHÄbAll_60hu¶™¶B
æ
˘void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float>, float>(float*, float*, int, tensorflow::functor::Sum<float>, float)0*Ä28Ä@ÄHÄb!conv2d_179/kernel/Regularizer/Sumhu  »B
æ
˘void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float>, float>(float*, float*, int, tensorflow::functor::Sum<float>, float)0*Ä28Ä@ÄHÄb!conv2d_182/kernel/Regularizer/Sumhu  »B
æ
˘void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float>, float>(float*, float*, int, tensorflow::functor::Sum<float>, float)0*Ä28Ä@ÄHÄb!conv2d_184/kernel/Regularizer/Sumhu  »B
æ
˘void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float>, float>(float*, float*, int, tensorflow::functor::Sum<float>, float)0*Ä28Ä@ÄHÄb!conv2d_185/kernel/Regularizer/Sumhu  »B
æ
˘void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float>, float>(float*, float*, int, tensorflow::functor::Sum<float>, float)0*Ä28Ä@ÄHÄb!conv2d_188/kernel/Regularizer/Sumhu  »B
æ
˘void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float>, float>(float*, float*, int, tensorflow::functor::Sum<float>, float)0*Ä28Ä@ÄHÄb!conv2d_190/kernel/Regularizer/Sumhu  »B
æ
˘void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float>, float>(float*, float*, int, tensorflow::functor::Sum<float>, float)0*Ä28Ä@ÄHÄb!conv2d_191/kernel/Regularizer/Sumhu  »B
º
˘void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float>, float>(float*, float*, int, tensorflow::functor::Sum<float>, float)0*Ä28Ä@ÄHÄbdense_21/kernel/Regularizer/Sumhu  »B
º
˘void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, int, tensorflow::functor::Sum<float>, float>(float*, float*, int, tensorflow::functor::Sum<float>, float)0*Ä28Ä@ÄHÄbdense_22/kernel/Regularizer/Sumhu  »B
ﬂ
§void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*) *Ä28Ä@ÄHÄXbmodel_7/dense_23/MatMulhu  »B
•
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Ä@ÄHÄbmodel_7/dense_21/BiasAddhuZUÖB
•
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2R8Ä@ÄHÄbmodel_7/dense_22/BiasAddhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb<cond_1/then/_10/cond_1/Adam/Adam/update_11/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb<cond_1/then/_10/cond_1/Adam/Adam/update_14/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb<cond_1/then/_10/cond_1/Adam/Adam/update_17/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb<cond_1/then/_10/cond_1/Adam/Adam/update_21/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb<cond_1/then/_10/cond_1/Adam/Adam/update_22/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb<cond_1/then/_10/cond_1/Adam/Adam/update_31/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb<cond_1/then/_10/cond_1/Adam/Adam/update_33/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb<cond_1/then/_10/cond_1/Adam/Adam/update_39/ResourceApplyAdamhuZUÖB
í
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb;cond_1/then/_10/cond_1/Adam/Adam/update_5/ResourceApplyAdamhuZUÖB
í
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb;cond_1/then/_10/cond_1/Adam/Adam/update_9/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb<cond_1/then/_10/cond_1/Adam/Adam/update_46/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb<cond_1/then/_10/cond_1/Adam/Adam/update_47/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb<cond_1/then/_10/cond_1/Adam/Adam/update_59/ResourceApplyAdamhuZUÖB
ì
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *Ä28Ä@ÄHÄb<cond_1/then/_10/cond_1/Adam/Adam/update_63/ResourceApplyAdamhuZUÖB
’
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAllhu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_14hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_15hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_17hu  »B
◊
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_2hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_22hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_23hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_25hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_27hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_29hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_30hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_31hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_33hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_37hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_38hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_39hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_41hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_45hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_53hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_58hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_65hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_66hu  »B
◊
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Ä@ÄHÄbAll_7hu  »B
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä2 8Ä@ÄHÄbAll_64hu  »B
◊
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä2 8Ä@ÄHÄbAll_8hu  »B
Î
¬void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*Ä28Ä@ÄHÄbSum_2hu  »B
á
¬void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*Ä28Ä@ÄHÄb!conv2d_171/kernel/Regularizer/Sumhu  »B
á
¬void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*Ä28Ä@ÄHÄb!conv2d_172/kernel/Regularizer/Sumhu  »B
á
¬void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*Ä28Ä@ÄHÄb!conv2d_173/kernel/Regularizer/Sumhu  »B
á
¬void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*Ä2 8Ä@ÄHÄb!conv2d_174/kernel/Regularizer/Sumhu  »B
á
¬void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*Ä2 8Ä@ÄHÄb!conv2d_177/kernel/Regularizer/Sumhu  »B
á
¬void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*Ä2 8Ä@ÄHÄb!conv2d_183/kernel/Regularizer/Sumhu  »B
â
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2 8Ä@ÄHÄb4gradient_tape/model_7/conv2d_171/BiasAdd/BiasAddGradhuZUÖB
â
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2 8Ä@ÄHÄb4gradient_tape/model_7/conv2d_172/BiasAdd/BiasAddGradhuZUÖB
â
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2@8Ä@ÄHÄb4gradient_tape/model_7/conv2d_173/BiasAdd/BiasAddGradhuZUÖB
ä
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Ä@ÄHÄb4gradient_tape/model_7/conv2d_176/BiasAdd/BiasAddGradhuZUÖB
ä
¥void tensorflow::functor::CleanupSegments<Eigen::half*, Eigen::half*, cub::Sum>(Eigen::half*, Eigen::half*, int, int, int, cub::Sum, std::iterator_traits<Eigen::half*>::value_type)*  2Ä8Ä@ÄHÄb4gradient_tape/model_7/conv2d_180/BiasAdd/BiasAddGradhuZUÖB
ÿ
±void tensorflow::functor::CleanupSegments<bool*, bool*, tensorflow::functor::And>(bool*, bool*, int, int, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type)* 28Ä@ÄHÄbAll_16huMUB
ÿ
±void tensorflow::functor::CleanupSegments<bool*, bool*, tensorflow::functor::And>(bool*, bool*, int, int, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type)* 28Ä@ÄHÄbAll_32huMUB
á
≈void tensorflow::functor::CleanupSegments<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)* 28Ä@ÄHÄb!conv2d_180/kernel/Regularizer/SumhuMUB
á
≈void tensorflow::functor::CleanupSegments<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)* 28Ä@ÄHÄb!conv2d_183/kernel/Regularizer/SumhuMUB
ö
ƒvoid tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿*  2@8Ä@ÄHÄb2gradient_tape/model_7/dense_21/BiasAdd/BiasAddGradhuZUÖB
©
√void tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)Ä!*  2@8Ä@ÄHÄbBgradient_tape/model_7/batch_normalization_78/batchnorm/mul_1/Sum_1huZUÖB
©
√void tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)Ä!*  2@8Ä@ÄHÄbBgradient_tape/model_7/batch_normalization_79/batchnorm/add_1/Sum_1huZUÖB
–
Åvoid tensorflow::functor::ColumnReduceKernel<float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, tensorflow::functor::Sum<float> >(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)Ä!*  2@8Ä@ÄHÄb+model_7/batch_normalization_78/moments/meanhuZUÖB
‘
Åvoid tensorflow::functor::ColumnReduceKernel<float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, tensorflow::functor::Sum<float> >(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)Ä!*  2@8Ä@ÄHÄb/model_7/batch_normalization_78/moments/variancehuZUÖB
¶
–void tensorflow::functor::ColumnReduceMax16ColumnsKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)¿* 28Ä@ÄHÄb2gradient_tape/model_7/dense_23/BiasAdd/BiasAddGradhu Ä¢B
ñ
◊void tensorflow::functor::RowReduceKernel<cub::TransformInputIterator<float, tensorflow::(anonymous namespace)::SubtractAndExpFunctor<Eigen::half, float>, cub::CountingInputIterator<int, long>, long>, float*, cub::Sum>(cub::TransformInputIterator<float, tensorflow::(anonymous namespace)::SubtractAndExpFunctor<Eigen::half, float>, cub::CountingInputIterator<int, long>, long>, float*, int, int, cub::Sum, std::iterator_traits<cub::TransformInputIterator<float, tensorflow::(anonymous namespace)::SubtractAndExpFunctor<Eigen::half, float>, cub::CountingInputIterator<int, long>, long> >::value_type)*Ä28Ä@ÄHÄbmodel_7/activation_87/Softmaxhu  »B
‘
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä28Ä@ÄHÄXbmodel_7/conv2d_173/Conv2DhuZUÖB
˜
óvoid tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 2, 0, 1, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*Ä28Ä@ÄHÄXb<gradient_tape/model_7/conv2d_173/Conv2D/Conv2DBackpropFilterhuZUÖB
]
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2 8Å@ÅHÅbmodel_7/dropout_47/dropout/MulhuZUÖB
∞
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Å@ÅHÅbmodel_7/conv2d_189/BiasAdd/CasthuZUÖB
∞
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Å@ÅHÅbmodel_7/conv2d_190/BiasAdd/CasthuZUÖB
µ
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Å@ÅHÅb0gradient_tape/model_7/dense_21/BiasAdd/Cast/CasthuZUÖB
®
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Å@ÅHÅb#model_7/batch_normalization_78/CasthuZUÖB
Æ
”void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Å@ÅHÅb9gradient_tape/model_7/batch_normalization_79/moments/CasthuZUÖB
ÿ
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Å@ÅHÅbAll_19hu  »B
◊
Ævoid tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *Ä28Å@ÅHÅbAll_9hu  »B
o
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb,model_7/batch_normalization_78/batchnorm/addhuZUÖB
b
"AddV2_GPU_DT_INT64_DT_INT64_kernel*Ä28Ä@ÄHÄbcond_1/then/_10/cond_1/Adam/addhuZUÖB
G
!Equal_GPU_DT_INT64_DT_BOOL_kernel*Ä28Ä@ÄHÄbEqualhuZUÖB
o
'GreaterEqual_GPU_DT_HALF_DT_BOOL_kernel*Ä2 8Ä@ÄHÄb'model_7/dropout_47/dropout/GreaterEqualhuZUÖB
g
(GreaterEqual_GPU_DT_INT64_DT_BOOL_kernel*Ä28Ä@ÄHÄbcond/then/_0/cond/GreaterEqualhuZUÖB
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
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_19hu  »B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄb
IsFinite_2hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_22hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_23hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_25hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_30hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_33hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_38hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_39hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_53hu  »B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄb
IsFinite_6hu  »B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄb
IsFinite_9hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_58hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_59hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_61hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_62hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_54hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_55hu  »B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄbIsFinite_64hu  »B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*Ä28Ä@ÄHÄb
IsFinite_8hu  »B
P
%LogicalAnd_GPU_DT_BOOL_DT_BOOL_kernel*Ä28Ä@ÄHÄb
LogicalAndhuZUÖB
D
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbMulhuZUÖB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb!conv2d_171/kernel/Regularizer/mulhuZUÖB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb!conv2d_172/kernel/Regularizer/mulhuZUÖB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb!conv2d_173/kernel/Regularizer/mulhuZUÖB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb!conv2d_176/kernel/Regularizer/mulhuZUÖB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb!conv2d_177/kernel/Regularizer/mulhuZUÖB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb!conv2d_178/kernel/Regularizer/mulhuZUÖB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb!conv2d_179/kernel/Regularizer/mulhuZUÖB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb!conv2d_180/kernel/Regularizer/mulhuZUÖB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb!conv2d_181/kernel/Regularizer/mulhuZUÖB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb!conv2d_184/kernel/Regularizer/mulhuZUÖB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb!conv2d_185/kernel/Regularizer/mulhuZUÖB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb!conv2d_186/kernel/Regularizer/mulhuZUÖB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb!conv2d_187/kernel/Regularizer/mulhuZUÖB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb!conv2d_189/kernel/Regularizer/mulhuZUÖB
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb!conv2d_190/kernel/Regularizer/mulhuZUÖB
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbdense_22/kernel/Regularizer/mulhuZUÖB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb/gradient_tape/conv2d_171/kernel/Regularizer/MulhuZUÖB
v
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb5gradient_tape/conv2d_171/kernel/Regularizer/mul/Mul_1huZUÖB
Å
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb@gradient_tape/model_7/batch_normalization_79/batchnorm/mul/Mul_1huZUÖB
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb2model_7/batch_normalization_78/AssignMovingAvg/mulhuZUÖB
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb2model_7/batch_normalization_79/AssignMovingAvg/mulhuZUÖB
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb4model_7/batch_normalization_79/AssignMovingAvg_1/mulhuZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_23huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_25huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_27huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_29huZUÖB
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_3huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_31huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_33huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_37huZUÖB
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_4huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_43huZUÖB
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_5huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_51huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_57huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_61huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_63huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_65huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_67huZUÖB
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_7huZUÖB
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_9huZUÖB
r
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb1gradient_tape/conv2d_172/kernel/Regularizer/Mul_1huZUÖB

 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb>gradient_tape/model_7/batch_normalization_78/batchnorm/mul/MulhuZUÖB
Å
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb@gradient_tape/model_7/batch_normalization_78/batchnorm/mul_2/MulhuZUÖB
É
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbBgradient_tape/model_7/batch_normalization_78/batchnorm/mul_2/Mul_1huZUÖB
Å
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb@gradient_tape/model_7/batch_normalization_79/batchnorm/mul_2/MulhuZUÖB
É
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbBgradient_tape/model_7/batch_normalization_79/batchnorm/mul_2/Mul_1huZUÖB
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb.model_7/batch_normalization_78/batchnorm/mul_2huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_60huZUÖB
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb/gradient_tape/conv2d_173/kernel/Regularizer/MulhuZUÖB
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_6huZUÖB
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄbmul_66huZUÖB
r
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb1gradient_tape/conv2d_174/kernel/Regularizer/Mul_1huZUÖB
]
Mul_GPU_DT_HALF_DT_HALF_kernel*Ä2 8Ä@ÄHÄbmodel_7/dropout_46/dropout/MulhuZUÖB

 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb>gradient_tape/model_7/batch_normalization_78/batchnorm/sub/Neghu  »B

 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb>gradient_tape/model_7/batch_normalization_79/batchnorm/sub/Neghu  »B
b
 Pow_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb!cond_1/then/_10/cond_1/Adam/Pow_1huZUÖB
q
"Rsqrt_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb.model_7/batch_normalization_78/batchnorm/Rsqrthu  »B
h
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb$conv2d_171/kernel/Regularizer/SquarehuZUÖB
h
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb$conv2d_173/kernel/Regularizer/SquarehuZUÖB
u
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb4model_7/batch_normalization_79/AssignMovingAvg_1/subhuZUÖB
m
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*Ä28Ä@ÄHÄb,model_7/batch_normalization_79/batchnorm/subhuZUÖB
µ
htensorflow::functor::ReluGradHalfKernelVector(Eigen::half const*, Eigen::half const*, Eigen::half*, int)*Ä2 8Ä@ÄHÄb,gradient_tape/model_7/activation_86/ReluGradhu  »B
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
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_7/conv2d_171/BiasAdd/CasthuZUÖB
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_7/conv2d_171/Conv2D/CasthuZUÖB
∞
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_7/conv2d_172/BiasAdd/CasthuZUÖB
∞
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_7/conv2d_173/BiasAdd/CasthuZUÖB
∞
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_7/conv2d_174/BiasAdd/CasthuZUÖB
∞
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_7/conv2d_176/BiasAdd/CasthuZUÖB
∞
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_7/conv2d_177/BiasAdd/CasthuZUÖB
∞
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_7/conv2d_178/BiasAdd/CasthuZUÖB
∞
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_7/conv2d_179/BiasAdd/CasthuZUÖB
∞
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_7/conv2d_180/BiasAdd/CasthuZUÖB
∞
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_7/conv2d_182/BiasAdd/CasthuZUÖB
∞
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_7/conv2d_183/BiasAdd/CasthuZUÖB
∞
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_7/conv2d_184/BiasAdd/CasthuZUÖB
∞
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_7/conv2d_185/BiasAdd/CasthuZUÖB
∞
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_7/conv2d_187/BiasAdd/CasthuZUÖB
∞
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_7/conv2d_188/BiasAdd/CasthuZUÖB
Æ
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_7/dense_23/BiasAdd/CasthuZUÖB
Æ
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_7/dense_21/BiasAdd/CasthuZUÖB
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_7/conv2d_173/Conv2D/CasthuZUÖB
≠
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_7/dense_23/MatMul/CasthuZUÖB
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbmodel_7/conv2d_174/Conv2D/CasthuZUÖB
Ø
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2@8Ä@ÄHÄbmodel_7/conv2d_177/Conv2D/CasthuZUÖB
«
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä@ÄHÄb6gradient_tape/model_7/batch_normalization_78/Cast/CasthuZUÖB
«
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä@ÄHÄb6gradient_tape/model_7/batch_normalization_79/Cast/CasthuZUÖB
‡
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä28Ä@ÄHÄb;gradient_tape/categorical_crossentropy/weighted_loss/Tile_1huZUÖB
Ò
≈void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb
div_no_nanhuZUÖB
Ω
€void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_rsqrt_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_rsqrt_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb@gradient_tape/model_7/batch_normalization_79/batchnorm/RsqrtGradhuZUÖB
¬
ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_inverse_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_inverse_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*Ä28Ä@ÄHÄbtruedivhuZUÖB
ã
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbCast_1huZUÖB
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
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb1gradient_tape/model_7/conv2d_171/Conv2D/Cast/CasthuZUÖB
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb1gradient_tape/model_7/conv2d_172/Conv2D/Cast/CasthuZUÖB
∑
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb2gradient_tape/model_7/conv2d_173/BiasAdd/Cast/CasthuZUÖB
∑
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb2gradient_tape/model_7/conv2d_174/BiasAdd/Cast/CasthuZUÖB
∑
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb2gradient_tape/model_7/conv2d_176/BiasAdd/Cast/CasthuZUÖB
∑
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb2gradient_tape/model_7/conv2d_177/BiasAdd/Cast/CasthuZUÖB
∑
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb2gradient_tape/model_7/conv2d_178/BiasAdd/Cast/CasthuZUÖB
∑
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb2gradient_tape/model_7/conv2d_179/BiasAdd/Cast/CasthuZUÖB
∑
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb2gradient_tape/model_7/conv2d_181/BiasAdd/Cast/CasthuZUÖB
∑
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb2gradient_tape/model_7/conv2d_183/BiasAdd/Cast/CasthuZUÖB
∑
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb2gradient_tape/model_7/conv2d_190/BiasAdd/Cast/CasthuZUÖB
∑
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb2gradient_tape/model_7/conv2d_191/BiasAdd/Cast/CasthuZUÖB
µ
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb0gradient_tape/model_7/dense_23/BiasAdd/Cast/CasthuZUÖB
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb1gradient_tape/model_7/conv2d_173/Conv2D/Cast/CasthuZUÖB
¥
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb/gradient_tape/model_7/dense_23/MatMul/Cast/CasthuZUÖB
∂
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄb1gradient_tape/model_7/conv2d_174/Conv2D/Cast/CasthuZUÖB
®
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä2{8Ä@ÄHÄb#model_7/batch_normalization_79/CasthuZUÖB
˝
’void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*Ä28Ä@ÄHÄbCast_7huZUÖB