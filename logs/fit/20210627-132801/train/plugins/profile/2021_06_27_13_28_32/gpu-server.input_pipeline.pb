	???x?y@???x?y@!???x?y@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC???x?y@??w?-`R@1?XR??Pt@A??%?ɦ?I??!??`??rEagerKernelExecute 0*	n??f?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?u?B?F@!?m??X@)?u?B?F@1?m??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?X?|^??!??B?@??)?X?|^??1??B?@??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismw???閝?!??aHf??)??]?p??1?????:Preprocessing2F
Iterator::Model??ܠ?!???а??)?N???p?1?~?HDT??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?g???F@!>??S?X@)(??ȯ_?1 ???Տq?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 18.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIhژ? ?2@Qf??7OT@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??w?-`R@??w?-`R@!??w?-`R@      ??!       "	?XR??Pt@?XR??Pt@!?XR??Pt@*      ??!       2	??%?ɦ???%?ɦ?!??%?ɦ?:	??!??`????!??`??!??!??`??B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qhژ? ?2@yf??7OT@