	)?k{??v@)?k{??v@!)?k{??v@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC)?k{??v@??}?ua@1???cZ?k@A<??fԜ?I9
? @rEagerKernelExecute 0*	?I2?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generatorѱ?J\?4@!????X@)ѱ?J\?4@1????X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?g?????!?`??%`??)?g?????1?`??%`??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism???o????!??|????)?<?E~???1|l???:Preprocessing2F
Iterator::Model?,????!?	?c???)ҏ?S??k?1Cq"jtݐ?:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap???ɣ4@!{N?w?X@)\;Qi[?1Ƹ?	f???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 37.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI? ?D?:C@Q`?f?@?N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??}?ua@??}?ua@!??}?ua@      ??!       "	???cZ?k@???cZ?k@!???cZ?k@*      ??!       2	<??fԜ?<??fԜ?!<??fԜ?:	9
? @9
? @!9
? @B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q? ?D?:C@y`?f?@?N@