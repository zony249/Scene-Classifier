	LnYk_?@LnYk_?@!LnYk_?@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCLnYk_?@???(?Ys@1??A??n@A??}"??IpxADj? @rEagerKernelExecute 0*	ˡE????@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator??d??B<@!??Y??X@)??d??B<@1??Y??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchG???R{??!@;:+????)G???R{??1@;:+????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??Kǜ?!????m??)????????1??????:Preprocessing2F
Iterator::ModelR}?%???!??1?01??)?aۢ?i?1?hO????:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?&OYMC<@!??ų??X@)G仔?d\?1j???|y?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 55.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?F??	L@Q??&?E@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???(?Ys@???(?Ys@!???(?Ys@      ??!       "	??A??n@??A??n@!??A??n@*      ??!       2	??}"????}"??!??}"??:	pxADj? @pxADj? @!pxADj? @B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?F??	L@y??&?E@