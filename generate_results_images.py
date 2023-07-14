
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def generate_results_from_path(dataset_path):
    plt.rcParams["figure.figsize"] = (18, 5)
    plt.rcParams["font.family"] = "Times New Roman"

    fig, axeslist = plt.subplots(ncols=6, nrows=2)
    axeslist[0,0].imshow(mpimg.imread(dataset_path + "src.png"))
    axeslist[0,0].get_xaxis().set_ticks([])
    axeslist[0,0].get_yaxis().set_ticks([])
    axeslist[0,0].set_ylabel("Input", rotation='horizontal', ha='right', fontsize=15, fontweight='bold')
    axeslist[1,0].set_axis_off()

    # First row
    axeslist[0,1].imshow(mpimg.imread(dataset_path + "gt_01.png"))
    axeslist[0,1].get_xaxis().set_ticks([])
    axeslist[0,1].get_yaxis().set_ticks([])
    axeslist[0,1].set_ylabel("GT", rotation='horizontal', ha='right', fontsize=15, fontweight='bold')

    axeslist[0,2].imshow(mpimg.imread(dataset_path + "gt_02.png"))
    axeslist[0,2].get_xaxis().set_ticks([])
    axeslist[0,2].get_yaxis().set_ticks([])

    axeslist[0,3].imshow(mpimg.imread(dataset_path + "gt_03.png"))
    axeslist[0,3].get_xaxis().set_ticks([])
    axeslist[0,3].get_yaxis().set_ticks([])

    axeslist[0,4].imshow(mpimg.imread(dataset_path + "gt_04.png"))
    axeslist[0,4].get_xaxis().set_ticks([])
    axeslist[0,4].get_yaxis().set_ticks([])

    axeslist[0,5].imshow(mpimg.imread(dataset_path + "gt_05.png"))
    axeslist[0,5].get_xaxis().set_ticks([])
    axeslist[0,5].get_yaxis().set_ticks([])

    # Second row
    axeslist[1,1].imshow(mpimg.imread(dataset_path + "predict_01.png"))
    axeslist[1,1].get_xaxis().set_ticks([])
    axeslist[1,1].get_yaxis().set_ticks([])
    axeslist[1,1].set_ylabel("Pred", rotation='horizontal', ha='right', fontsize=15, fontweight='bold')

    axeslist[1,2].imshow(mpimg.imread(dataset_path + "predict_02.png"))
    axeslist[1,2].get_xaxis().set_ticks([])
    axeslist[1,2].get_yaxis().set_ticks([])

    axeslist[1,3].imshow(mpimg.imread(dataset_path + "predict_03.png"))
    axeslist[1,3].get_xaxis().set_ticks([])
    axeslist[1,3].get_yaxis().set_ticks([])

    axeslist[1,4].imshow(mpimg.imread(dataset_path + "predict_04.png"))
    axeslist[1,4].get_xaxis().set_ticks([])
    axeslist[1,4].get_yaxis().set_ticks([])

    axeslist[1,5].imshow(mpimg.imread(dataset_path + "predict_05.png"))
    axeslist[1,5].get_xaxis().set_ticks([])
    axeslist[1,5].get_yaxis().set_ticks([])

    # fig.subplots_adjust(hspace=0.0)
    fig.tight_layout()
    plt.savefig(dataset_path + 'comparative.png', format='png')
    # plt.show()


for i in range (0, 10):
    
    dataset_path = '/home/aiiacvmllab/Projects/nvs_repos/Look-Outside-Room/experiments/custom/exp_multitask_refinery/2023-07-13_12:43:07_evaluate_frame_6_video_10_gap_35/00'
    generate_results_from_path(dataset_path + str(i) + "/")
