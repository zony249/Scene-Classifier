	??1˧x@??1˧x@!??1˧x@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC??1˧x@??#???b@1ak???Dn@A1@?	???Ikc섗???rEagerKernelExecute 0*	?$??2?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratorN?»\?3@!??	Ȇ?X@)N?»\?3@1??	Ȇ?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchm???5??!?"?=س?)m???5??1?"?=س?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism????EB??!?G??*U??)Ͽ]??N??1&?(0???:Preprocessing2F
Iterator::Model??%VF#??!?b????)?~?o?1L??H<???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?EР3@!O????X@)?8?Վ?\?1????]??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 38.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI???$OC@Qx	 ?۰N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??#???b@??#???b@!??#???b@      ??!       "	ak???Dn@ak???Dn@!ak???Dn@*      ??!       2	1@?	???1@?	???!1@?	???:	kc섗???kc섗???!kc섗???B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q???$OC@yx	 ?۰N@