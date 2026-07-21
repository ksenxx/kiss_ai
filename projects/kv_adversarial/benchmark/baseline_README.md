# Put your engine here

Drop your implementation as `baseline/*.cc` — it must define:

    IKVStore* create_kvstore();

(interface in `../include/kvstore_interface.h`). Then `../build.sh` compiles
`harness/ + baseline/*.cc` into `kvstore_bench`. C++17 + pthread only, no external deps.
