#! /usr/bin/python3

import subprocess
import argparse


def get_PRs():
    PRs = []

    PRs.append([
        "google_upstream",
        "35394",
        "[ROCm] Support of GRU and LSTM",
        "OPEN",
        "None",
        [
            ("ae0e325a9fd53f2981bc569a2e3f8699c72a2ddc",["3ebe6b2479bd0578953a18298ff55863fa20ed61"]),
            ("451487e515d978aeec3c7e3c8f42ad11e51ac0ab",["ae0e325a9fd53f2981bc569a2e3f8699c72a2ddc", "6deb2d610d7671ace9f5a5ddc8c61c30dca72665"]),
        ]])

    PRs.append([
        "google_upstream",
        "35395",
        "[ROCm] stateful random ops",
        "MERGED",
        "b84908da7797139892c68c7680375dd6389c71a0",
        [
            ("968a674ecb6db34e5d2e09068a8d9ca5ca4e3e24",["a1aa33532810dc16c8338afe6649cbd4ec1787f9"]),
            ("f7b28191777b6ae86c0dbdab7a74b8370e53eaa8",["968a674ecb6db34e5d2e09068a8d9ca5ca4e3e24"]),
            ("eee5851777b842945b12937600b005a58aae0f2c",["f7b28191777b6ae86c0dbdab7a74b8370e53eaa8"]),
            ("3122804c8537e920c4cfd2b92c141fd400fabe19",["eee5851777b842945b12937600b005a58aae0f2c"]),
            ("b84908da7797139892c68c7680375dd6389c71a0",["3122804c8537e920c4cfd2b92c141fd400fabe19"]),
        ]])

    PRs.append([
        "google_upstream",
        "35519",
        "[ROCm] Unit-test updates for the ROCm platform.",
        "OPEN",
        "None",
        [
            ("035de975e6b21cadc09ad84e083d27bd025b8139",["850066a62d4d3af87e042a60a91ed3531aba66c5"]),
        ]])

    PRs.append([
        "google_upstream",
        "35624",
        "[ROCm] Misc updates for the ROCm platform",
        "MERGED",
        "04fb568df083a1903dd4f061539b29b4a849fd18",
        [
            ("09e2eaf34227ef922d8e85b0caef2c0eb5749df5",["ebc6577281722dcf02c2a7cecbe368bdd8ddea4d"]),
            ("d325b255ff7d0bf1ca04229880dffb0a37d52e2d",["09e2eaf34227ef922d8e85b0caef2c0eb5749df5"]),
            ("17b87f0b51ad290269f983a85b887ae838c2ebe2",["d325b255ff7d0bf1ca04229880dffb0a37d52e2d"]),
            ("04fb568df083a1903dd4f061539b29b4a849fd18",["17b87f0b51ad290269f983a85b887ae838c2ebe2"]),
        ]])

    PRs.append([
        "google_upstream",
        "35666",
        "[ROCm] Support for complex type BLAS operations",
        "OPEN",
        "None",
        [
            ("c329f1c5020c3df814be0a1e98cd740c5a4e4621",["77994fbf12c37d09952e8a34381129534dd572af"]),
            ("543db6fc6713ed9ba19cf798a92f4bd2f4ad9ba2",["c329f1c5020c3df814be0a1e98cd740c5a4e4621"]),
            ("0f0aa375122dea501a237a5f8462bfde31d03a7d",["543db6fc6713ed9ba19cf798a92f4bd2f4ad9ba2"]),
            ("b64dde60e829387ce81bf88d0cb23402a9211245",["0f0aa375122dea501a237a5f8462bfde31d03a7d", "8e65693af70c044fd71f5bf918854d83a64ba51c"]),
        ]])

    PRs.append([
        "google_upstream",
        "35752",
        "[ROCm] Fix the ROCm CSB breakage - 200110",
        "OPEN",
        "None",
        [
            ("f8ba03dfd976278a605e53ac741210fbab14c7ae",["7f216cffbadb55a0e310c51692c71b2add261bae"]),
        ]])

    PRs.append([
        "google_upstream",
        "35890",
        "[ROCm] Fix for //tensorflow/python/kernel_tests/signal:mel_ops_test",
        "OPEN",
        "None",
        [
            ("3c7596fc76855b07125cbfa9eb2a7a093ee8719c",["1ceda293f6041b95673c0dc5ab3b5b30de15e536"]),
            ("c573019702d99e412bc9cdc016a4ac44a7f36a9c",["3c7596fc76855b07125cbfa9eb2a7a093ee8719c"]),
        ]])

    PRs.append([
        "google_upstream",
        "35891",
        "[ROCm] Fix for //tensorflow/python/eager/benchmarks/resnet50:*",
        "MERGED",
        "ee1c2f112dc666a19e0d9e4ad679c0e948ebde8a",
        [
            ("ee1c2f112dc666a19e0d9e4ad679c0e948ebde8a",["1ceda293f6041b95673c0dc5ab3b5b30de15e536"]),
        ]])

    PRs.append([
        "google_upstream",
        "35924",
        "[ROCm] Optimized training apply ops",
        "OPEN",
        "None",
        [
            ("3693256ba03f67d38fe33daa495b296d45539076",["057cf24986e452e56ddcc86e4366c8adfd1127f0"]),
            ("fb91bf645837c5755daedfe537a1acf8f5aa8f65",["3693256ba03f67d38fe33daa495b296d45539076"]),
            ("9760afc11963073c805156ab7acd4e32898c34a4",["fb91bf645837c5755daedfe537a1acf8f5aa8f65", "4221989d4952d65bfbd343f2d93a00c323348151"]),
        ]])

    PRs.append([
        "google_upstream",
        "35964",
        "[ROCm] Enabling //tensorflow/core/framework:variant_op_registry_test",
        "MERGED",
        "c1b02f9a649cdd0793e667ccb7445d50e5095ceb",
        [
            ("c1b02f9a649cdd0793e667ccb7445d50e5095ceb",["057cf24986e452e56ddcc86e4366c8adfd1127f0"]),
        ]])

    PRs.append([
        "google_upstream",
        "35965",
        "[ROCm] Enabling the kernel Relu for float16",
        "OPEN",
        "None",
        [
            ("1087b24938690a5d747a6e51e6d00a10d039e898",["db8a74a737cc735bb2a4800731d21f2de6d04961"]),
        ]])

    PRs.append([
        "google_upstream",
        "35966",
        "[ROCm] Complex sparse ops",
        "OPEN",
        "None",
        [
            ("815fa1866caa88727d2d5288c028f0e610c4bda3",["db8a74a737cc735bb2a4800731d21f2de6d04961"]),
            ("15e47e640272ec89ec76098354707969dbcf3c67",["815fa1866caa88727d2d5288c028f0e610c4bda3"]),
        ]])

    PRs.append([
        "google_upstream",
        "35971",
        "[ROCm] GpuManagedAllocator and common_runtime unit tests",
        "OPEN",
        "None",
        [
            ("af54994072bda083229fd11cb2b1d58e2cd38ab0",["db8a74a737cc735bb2a4800731d21f2de6d04961"]),
        ]])

    PRs.append([
        "google_upstream",
        "35972",
        "[ROCm] Enabling several cwise op kernels",
        "MERGED",
        "d45988d245ada11dc2b1168e86f1d4441e0f844c",
        [
            ("d45988d245ada11dc2b1168e86f1d4441e0f844c",["db8a74a737cc735bb2a4800731d21f2de6d04961"]),
        ]])

    PRs.append([
        "google_upstream",
        "35991",
        "[ROCm][XLA] Enabling llvm compiler test",
        "MERGED",
        "b27daf6d733ec695b64ad6a66508aeead918dbcc",
        [
            ("dfd7758505713de339367486741e7c24d0693490",["1f542b7c69aade026ad28b45f70fd0c8795d2f35"]),
            ("b27daf6d733ec695b64ad6a66508aeead918dbcc",["dfd7758505713de339367486741e7c24d0693490"]),
        ]])

    PRs.append([
        "google_upstream",
        "36017",
        "[ROCm] Fixing a bug in the previous commit to nccl_manager_test.cc file. ",
        "MERGED",
        "cd1a77fd5e6438b825cc90bd34706a8edeea809a",
        [
            ("cd1a77fd5e6438b825cc90bd34706a8edeea809a",["2df7f0fd535e007f7f22791513e73334fec953ec"]),
        ]])

    PRs.append([
        "google_upstream",
        "36018",
        "[ROCm] Changing cuda* names to gpu* names in a couple of tests",
        "MERGED",
        "1a071541c184204fef1359e719586909de36f492",
        [
            ("1a071541c184204fef1359e719586909de36f492",["2df7f0fd535e007f7f22791513e73334fec953ec"]),
        ]])

    PRs.append([
        "google_upstream",
        "36019",
        "[ROCm] Adding/Removing no_rocm tag to/from tests",
        "MERGED",
        "ff523f35ced55fde5e6fd40df30873e5d806de6f",
        [
            ("ff523f35ced55fde5e6fd40df30873e5d806de6f",["13178c55c70f63b593ef282dcfe3734d4bc67e91"]),
        ]])

    PRs.append([
        "google_upstream",
        "36031",
        "[ROCm] Fix for compile error in //tensorflow/compiler/xla/service:dynamic_padder_test",
        "CLOSED",
        "None",
        [
            ("3a5862c887f2df5030ef741c35c5600c90569771",["2df7f0fd535e007f7f22791513e73334fec953ec"]),
        ]])

    PRs.append([
        "google_upstream",
        "36032",
        "[ROCm] Fix for compile error in //tensorflow/compiler/xla:refcounting_hash_map_test",
        "CLOSED",
        "None",
        [
            ("e6fa8e3e024d3c337124e977e64f1f63936ac5b7",["2df7f0fd535e007f7f22791513e73334fec953ec"]),
        ]])

    PRs.append([
        "google_upstream",
        "36106",
        "[ROCm] add ROCm support for XLA RCCL thunk",
        "OPEN",
        "None",
        [
            ("ab136158ca7b69f22f9225bae91887ed44d0e863",["0a67210d6eb498866adce785b68ca906330b4f3a"]),
            ("1206aa6ee8435a33bb5a729b5fc1bc7c6bd704a0",["ab136158ca7b69f22f9225bae91887ed44d0e863"]),
        ]])

    PRs.append([
        "google_upstream",
        "36110",
        "[ROCm][XLA] Adding address space cast in ir_emitter",
        "MERGED",
        "062e7530b24b77f34947a1631ed968356984d5a8",
        [
            ("a63f8eeb227b1f49e7f514b4ed8e7617c01d9013",["7b0cb335fbd1365645d71e488517a6a80d26dee0"]),
            ("062e7530b24b77f34947a1631ed968356984d5a8",["a63f8eeb227b1f49e7f514b4ed8e7617c01d9013"]),
        ]])

    PRs.append([
        "google_upstream",
        "36187",
        "[ROCm][XLA:GPU] Fixing Atomic CAS codegen in ir_emitter",
        "CLOSED",
        "None",
        [
            ("c636d349510b6b1c480d2320bcbb21b51b4a005a",["9d1ec06b01d741f258589ea9279a03b488551f9a"]),
        ]])

    PRs.append([
        "google_upstream",
        "36191",
        "[ROCm][XLA] Adding three passes to  amdgpu compiler",
        "OPEN",
        "None",
        [
            ("9d3d2d3b3e4d8ef139970460490e30c11a29540b",["9d1ec06b01d741f258589ea9279a03b488551f9a"]),
        ]])

    PRs.append([
        "google_upstream",
        "36263",
        "[ROCm] Adding ROCm support for CTC Loss",
        "MERGED",
        "d7dbb77bd663aaa016efc9cf350169f418b0905d",
        [
            ("d7dbb77bd663aaa016efc9cf350169f418b0905d",["e1a66cff303ebdbc25280b9ad6f749504ec95534"]),
        ]])

    PRs.append([
        "google_upstream",
        "36267",
        "[ROCm] Reverting ROCm to use MIOpen Find Mode APIs (be default) for convolution",
        "OPEN",
        "None",
        [
            ("5675e37e5f9b595dab45f44239cbfab222e9dcc2",["e0374aaae8d1d841bd85c2361fe5f7ab8dbbd79f"]),
            ("5fe0ad377dc7e333acf8aac91e3333781242fe5c",["5675e37e5f9b595dab45f44239cbfab222e9dcc2"]),
            ("e3dcc169353646c4b5e684b7398cf1db743079cb",["5fe0ad377dc7e333acf8aac91e3333781242fe5c"]),
            ("4d4a5cede3b6e959fcc06fdb6211e4c9ef5343f5",["e3dcc169353646c4b5e684b7398cf1db743079cb"]),
            ("64ffda476af322ad804d1f5b8d7a05719e2f183c",["4d4a5cede3b6e959fcc06fdb6211e4c9ef5343f5"]),
            ("d42a76e177a26124e966c83cbbb809dbdbdcabbe",["64ffda476af322ad804d1f5b8d7a05719e2f183c"]),
            ("416aeccbfc430c71b27cbe04a57dcd1577b34fae",["d42a76e177a26124e966c83cbbb809dbdbdcabbe"]),
            ("253664ce7ee59bb2ffbc2b4b3fe94963e54837c1",["416aeccbfc430c71b27cbe04a57dcd1577b34fae"]),
            ("30debc7b11afdbc1651c860b65cdd2fba1b9ba50",["253664ce7ee59bb2ffbc2b4b3fe94963e54837c1"]),
            ("b0b670e6ee2eaa6823618d4aa8858846a4cbbd89",["30debc7b11afdbc1651c860b65cdd2fba1b9ba50"]),
        ]])

    PRs.append([
        "google_upstream",
        "36292",
        "[XLA][ROCm] Disable test that invokes rocBlas TRSM",
        "MERGED",
        "1f4186c64f76854fe26335729022b7dea4dec941",
        [
            ("1f4186c64f76854fe26335729022b7dea4dec941",["e1a66cff303ebdbc25280b9ad6f749504ec95534"]),
        ]])

    PRs.append([
        "google_upstream",
        "36341",
        "[ROCm] Adding no_rocm tag to tests that started failing after the 200129 weekly sync",
        "OPEN",
        "None",
        [
            ("2c6fae63bda1e9aa2022000e8c70122ecc006eb0",["9c7d78bce12a6374748ce84dd9a4fc0a37606775"]),
        ]])

    PRs.append([
        "google_upstream",
        "36342",
        "[ROCm][XLA] Fixing binary_ops_test",
        "MERGED",
        "7295af2b443454b2e8e6e69a70b9b08a255e0dfc",
        [
            ("7295af2b443454b2e8e6e69a70b9b08a255e0dfc",["944e6fe82a2b7733dd2f58ad352fcaeb7ad066b8"]),
        ]])

    PRs.append([
        "google_upstream",
        "36351",
        "[ROCm][XLA] Disable ResizeBilinearTest in image_ops_test",
        "OPEN",
        "None",
        [
            ("ae6fbe68f01839fd19230f38e43edc6c4fc8ab08",["944e6fe82a2b7733dd2f58ad352fcaeb7ad066b8"]),
        ]])

    # PRs.append([
    #     "rocm_fork",
    #     "782",
    #     "[DO NOT MERGE] Porting ROCm Fusion support to master-rocm-enhanced",
    #     "OPEN",
    #     "None",
    #     [
    #         ("a1cbb7a7844cdb0dd82c0c7d15c871d00acb2c5d",["1384c138c75d204ba0be9049be2bfd063552ea33"]),
    #     ]])

    # PRs.append([
    #     "rocm_fork",
    #     "783",
    #     "[DO NOT MERGE] Porting ROCm blfoat16 support to master-rocm-enhanced",
    #     "OPEN",
    #     "None",
    #     [
    #         ("9bf87efe4d7ccbd00732767824af60f886d59c0e",["1384c138c75d204ba0be9049be2bfd063552ea33"]),
    #     ]])

    # PRs.append([
    #     "rocm_fork",
    #     "789",
    #     "[DO NOT MERGE] Porting ROCm hipclang support to mater-rocm-enhanced",
    #     "OPEN",
    #     "None",
    #     [
    #         ("afe52fc3212b97d811d143b173ba68d494e26592",["81fcd56fb3beff2c2c0d78173403d144d525be82"]),
    #     ]])

    # PRs.append([
    #     "rocm_fork",
    #     "790",
    #     "[DO NOT MERGE] Porting ROCm Dropout support to master-rocm-enhanced",
    #     "OPEN",
    #     "None",
    #     [
    #         ("1ed173b7374f2e590b624e0a7e4fd5434b56ab36",["1384c138c75d204ba0be9049be2bfd063552ea33"]),
    #         ("f205c5a7b957ea62672fd1cefbbc25264ec134da",["1ed173b7374f2e590b624e0a7e4fd5434b56ab36"]),
    #         ("1871ee82a8806fee4c3a7e11fc46c5a59f190651",["f205c5a7b957ea62672fd1cefbbc25264ec134da"]),
    #         ("c9b7778e0ba4dedc720bf067b77b5c8746cde74a",["1871ee82a8806fee4c3a7e11fc46c5a59f190651"]),
    #     ]])

    # PRs.append([
    #     "rocm_fork",
    #     "791",
    #     "[DO NOT MERGE] Porting ROCm githooks to the master-rocm-enhanced branch",
    #     "OPEN",
    #     "None",
    #     [
    #         ("9c8604f7d367a0e7f21bcd3536b634f666dc88f5",["81fcd56fb3beff2c2c0d78173403d144d525be82"]),
    #     ]])

    # PRs.append([
    #     "rocm_fork",
    #     "794",
    #     "[DO NOT MERGE] Porting ROCm docs to master-rocm-enhanced",
    #     "OPEN",
    #     "None",
    #     [
    #         ("8943b7a1e32ecc434bfe31c11c027b9a8fcc1147",["1384c138c75d204ba0be9049be2bfd063552ea33"]),
    #     ]])

    # PRs.append([
    #     "rocm_fork",
    #     "795",
    #     "[DO NOT MERGE] Porting ROCm scripts to master-rocm-enhanced branch",
    #     "OPEN",
    #     "None",
    #     [
    #         ("7f5e9c8d5118eb72e473fa60ceb71ed6efffed30",["1ceda293f6041b95673c0dc5ab3b5b30de15e536"]),
    #     ]])

    # PRs.append([
    #     "rocm_fork",
    #     "799",
    #     "[DO NOT MERGE] Porting ROCm batch_gemm support to master-rocm-enhanced",
    #     "OPEN",
    #     "None",
    #     [
    #         ("0e2126f1f74139c9c8e5c860e02cbbc4dae9fa3e",["1ceda293f6041b95673c0dc5ab3b5b30de15e536"]),
    #     ]])

    # PRs.append([
    #     "rocm_fork",
    #     "803",
    #     "[DO NOT MERGE] Porting 3d pooling support to master-rocm-enhanced",
    #     "OPEN",
    #     "None",
    #     [
    #         ("e479022fd61b611874bf5ed5db49a0e15a619e9a",["1384c138c75d204ba0be9049be2bfd063552ea33"]),
    #         ("af22602cdc993379e89f5cd32e9d2541996b7ab0",["e479022fd61b611874bf5ed5db49a0e15a619e9a"]),
    #     ]])
    
    return PRs


def get_commit_specific_options(commit):
    commit_specific_options = {}
    commit_specific_options["c329f1c5020c3df814be0a1e98cd740c5a4e4621"] = ["-Xtheirs"]
    return commit_specific_options.get(commit, [])


def check_for_merge_commit(commits):

    commits_from_this_PR = set();
    head_commit = None
    base_commit = None
    has_merge_commit = False
    
    for commit, parents in commits:
        head_commit = commit
        commits_from_this_PR.add(commit)
        if len(parents) > 1:
            has_merge_commit = True
        for parent in parents:
            if parent not in commits_from_this_PR:
                base_commit = parent

    return has_merge_commit, base_commit, head_commit
    

def cherry_pick_simple(number, title, commits):

    for commit, parents in commits:
        cherry_pick_cmd = ["git", "cherry-pick", "--no-commit", commit]
        cherry_pick_cmd.extend(get_commit_specific_options(commit))
        # print (cherry_pick_cmd)
        result = subprocess.run(cherry_pick_cmd)
        if result.returncode != 0:
            print ("...FAILED (cherry-pick --no-commit)")
            return result.returncode

    commit_cmd = ["git", "commit", "-m", "PR {} - {}".format(number, title)]
    result = subprocess.run(commit_cmd)
    if result.returncode != 0:
        print ("...FAILED (commit)")
        return result.returncode

    return result.returncode


def cherry_pick_diff_apply(number, title, base_commit, head_commit):

    diff_cmd = ["git", "diff", "{}...{}".format(base_commit, head_commit), "--color=never", "--output", "patch.diff"]
    result = subprocess.run(diff_cmd)
    if result.returncode != 0:
        print ("...FAILED (diff)")
        return result.returncode

    apply_cmd = ["git", "apply", "--whitespace=nowarn", "patch.diff"]
    result = subprocess.run(apply_cmd)
    if result.returncode != 0:
        print ("...FAILED (apply)")
        return result.returncode
        
    commit_cmd = ["git", "commit", "--all", "-m", "PR {} - {}".format(number, title)]
    result = subprocess.run(commit_cmd)
    if result.returncode != 0:
        print ("...FAILED (commit)")
        return result.returncode

    return result.returncode


def cherry_pick_PRs(dest_branch_base_commit):

    for repo, number, title, state, pr_merge_commit, commits in get_PRs():
        
        print ("Cherry-picking {} PR {} - {}".format(state, number, title))

        if state == "CLOSED":
            print ("...Skipping cherry pick as PR was CLOSED without merging")
            continue
            
        if state == "MERGED":
            assert (pr_merge_commit != "None")
            is_ancestor_cmd = ["git", "merge-base", "--is-ancestor", pr_merge_commit, dest_branch_base_commit]
            # print (is_ancestor_cmd)
            result = subprocess.run(is_ancestor_cmd)
            if result.returncode == 0:
                print ("...Skipping cherry pick as PR is ALREADY MERGED into the destination branch")
                continue

        has_merge_commit, base_commit, head_commit = check_for_merge_commit(commits)
        if (repo == "google_upstream") and has_merge_commit:
            print ("...using the DIFF_APPLY route to do the cherry-picking")
            result = cherry_pick_diff_apply(number, title, base_commit, head_commit)
        else:
            print ("...using the SIMPLE route to do the cherry-picking")
            result = cherry_pick_simple(number, title, commits)

        if result != 0:
            return
            
            
def cherry_pick_commits():

    commits = [
        "2294526571ca2bb959a616b3feceebef8e525b7c",
        "916584b2746064ba3096200c4129a56320124b1c",
        "72bae743a5a6d1ec03078b494df37f8c2ddd2b08",
        "900a3db2a5af1992dc5d771edbb0874afd5a3ad3",
        "236cb3ac9fc40567b0f2ac6d8752cdbd523735f2",
        "8e1987c2db551ad44cbb60f292bb89d8fe37f1dd",
        "7387909d1cf10143ef0e33e8fc4652846297ec0a",
        "8fc5646d0ce97fbef44727dcf36c1c9a6a672ee9",
        "67f97220a9a3be65fb4e7cddf6d32afdadb2297e",
        "80f8bb77c698f1bae573dcdb3329b2e4d8cd815c",
        ]


    for commit in commits:
        cherry_pick_cmd = ["git", "cherry-pick", commit]
        result = subprocess.run(cherry_pick_cmd)
        if result.returncode != 0:
            print ("...FAILED (cherry-pick {}".format(commit))
            return


if __name__ == "__main__" :

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--base_commit", required=True)
    # args = parser.parse_args()

    dest_branch_base_commit = "05ea2b6d7f7a986827d9c9ec32a4a94e30714f9c"
    cherry_pick_PRs(dest_branch_base_commit)
    # cherry_pick_PRs_alt()
    # cherry_pick_commits()
