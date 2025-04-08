def compute_gpu_values(lane_id):  
    # Constants  
    c0 = 0  
    c1 = 1  
    c2 = 2  
    c4 = 4  
    c8 = 8  
      
    # Lane-specific calculations  
    floordiv = lane_id // c4  
    remainder = lane_id % c4  
  
    # Compute %9  
    cmpi_result = remainder < c0  
    addi_result = remainder + c4  
    select_result_11 = addi_result if cmpi_result else remainder  
  
    # Compute %12  
    minsi_result_12 = min(floordiv, c2)  
  
    # Compute %14  
    subi_result_13 = c2 - minsi_result_12  
    minsi_result_14 = min(subi_result_13, c1)  
  
    # Compute %16  
    muli_result_15 = select_result_11 * c8  
    minsi_result_16 = min(muli_result_15, c4)  
  
    # Compute %18  
    subi_result_17 = c4 - minsi_result_16  
    minsi_result_18 = min(subi_result_17, c8)  
  
    # Compute %19 and %21  
    create_mask_19 = [True] * minsi_result_18 + [False] * (8 - minsi_result_18)  
    select_result_21 = create_mask_19 if minsi_result_14 > c0 else [False] * 8  
  
    return minsi_result_12, minsi_result_16, select_result_21  
  
def compute_new_gpu_values(lane_id):  
    # Constants  
    c0 = 0  
    c1 = 1  
    c2 = 2  
    c4 = 4  
    c8 = 8  
  
    # Lane-specific calculations  
    floordiv = lane_id // c2  
    remainder = lane_id % c2  
  
    # Compute %26  
    cmpi_result_26 = remainder < c0  
    addi_result_27 = remainder + c2  
    select_result_28 = addi_result_27 if cmpi_result_26 else remainder  
  
    # Compute %29  
    minsi_result_29 = min(floordiv, c4)  
  
    # Compute %31  
    subi_result_30 = c4 - minsi_result_29  
    minsi_result_31 = min(subi_result_30, c1)  
  
    # Compute %33  
    muli_result_32 = select_result_28 * c8  
    minsi_result_33 = min(muli_result_32, c2)  
  
    # Compute %35  
    subi_result_34 = c2 - minsi_result_33  
    minsi_result_35 = min(subi_result_34, c8)  
  
    # Compute %36 and %38  
    create_mask_36 = [True] * minsi_result_35 + [False] * (8 - minsi_result_35)  
    select_result_38 = create_mask_36 if minsi_result_31 > c0 else [False] * 8  
  
    # The vector load and selection  
    select_result_40 = select_result_38  # Simplifying this as vector load emulation  
  
    return minsi_result_29, minsi_result_33, select_result_40
  
def main():  
    for lane_id in range(64):  
        minsi_result_12, minsi_result_16, select_result_21 = compute_gpu_values(lane_id)  
        minsi_result_29, minsi_result_33, select_result_40 = compute_new_gpu_values(lane_id)  
        print(f"Lane ID {lane_id}: %12 = {minsi_result_12}, %16 = {minsi_result_16}, %21 = {select_result_21}")  
        print(f"Lane ID {lane_id}: %29 = {minsi_result_29}, %33 = {minsi_result_33}, %40 = {select_result_40}")  
  
if __name__ == "__main__":  
    main()  

