import os
import time
import torch
import sys
import matplotlib.pyplot as plt
import numpy as np
import argparse

sys.path.append(os.path.join(os.getcwd(), 'models'))

from attacks.LBA.LBA_dataset import get_LBA_Dataset, get_total_sensitive_point_from_adv
from attacks.LBA.LBA_model import train_model
from utils import get_Adv_from_csv, check_incorrect_cnt_of_NVITA, append_result_to_csv_file, \
    create_empty_result_csv_file, check_result_file_path, set_seed

from attacks.nvita import NVITA
from attacks.fgsm import FGSM
from attacks.bim import BIM
from attacks.LBA.LBA_model import CNN_LBA_Model
from data_provider import get_ori_data
from models.model_utils import load_model


def Non_LBA_attack(data, model, args, attack) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
    """
    Non-LBA attack, available for NVITA, FGSM, BIM
    :param data: the original dataset
    :param model: the model to be attacked for TSF
    :param args: parameters for the attack, using attack, epsilon, n, maxiter, tol, seed
    :param attack: the attack method, options: [NVITA, FGSM, BIM]
    :return: X_adv_total: the adversarial examples,
             Y_adv_total: the attacked prediction,
             Y_pred_total: the original prediction
    """
    X_adv_total = torch.empty(0)
    Y_adv_total = torch.empty(0)
    Y_pred_total = torch.empty(0)
    path_out_dir = None

    # get parameters from the args
    eps = args.epsilon
    n = args.n
    maxiter = args.maxiter
    total = args.tol
    seed = args.seed
    save_as_csv = args.save_csv
    print_info = args.print_info

    attack_name = attack
    if attack == 'NVITA':
        attack_name = f'Non_Targeted_{n}VITA'

    csv_name = f'df_{data.df_name}_seed_{data.seed}_model_{model}_epsilon_{eps}_attack_{attack_name}.csv'

    if save_as_csv is True:
        path_out_dir = os.path.join(ATK_PATH, attack)
        path_out_dir = check_result_file_path(os.path.join(path_out_dir, csv_name))
        first_result_line_list = ["df", "Seed", "Model", "Epsilon", "Test Index", "Attack Name", "True Y",
                                  "Original Y Pred", "Attacked Y Pred", "Attacked AE", "Original AE", "Max Per",
                                  "Sum Per", "Cost Time", "Window Range", "Adv Example"]
        create_empty_result_csv_file(path_out_dir, first_result_line_list)

    if print_info:
        print(f'Dataset:{data} Attack:{attack} Model:{model} Epsilon:{eps} Seed:{seed} '
              f'Test Count:{data.X_test.shape[0]}')

    model.to(Device)
    model.eval()
    for test_ind in range(data.X_test.shape[0]):
        X_current = data.X_test[test_ind].unsqueeze(0).to(Device)
        ground_truth_y = data.Y_test[test_ind].unsqueeze(0).to(Device)
        window_range = data.window_ranges[test_ind].to(Device)

        start_time = time.time()
        X_adv = None
        att = None
        if attack == 'NVITA':
            att = NVITA(n, eps, model, maxiter, total, targeted=False)
            X_adv, _ = att.attack(X_current, ground_truth_y, window_range, seed=seed)
        elif attack == 'FGSM':
            att = FGSM(eps, model)
            X_adv = att.attack(X_current, ground_truth_y, window_range)
        elif attack == 'BIM':
            step = 200
            att = BIM(eps, eps / step, step, model)
            X_adv = att.attack(X_current, ground_truth_y, window_range)

        cost_time = time.time() - start_time
        original_y_pred = model(X_current).item()
        original_ae = np.absolute(original_y_pred - ground_truth_y.item())
        adv_y_pred = model(X_current).item()
        attacked_ae = np.absolute(adv_y_pred - ground_truth_y.item())
        eta = X_adv - X_current
        sum_per = torch.sum(torch.abs(eta)).item()
        max_per = torch.max(torch.abs(eta)).item()

        X_adv_total = torch.cat((X_adv_total, X_adv.clone().to('cpu')), dim=0)
        Y_adv_total = torch.cat((Y_adv_total, torch.tensor([adv_y_pred])), dim=0)
        Y_pred_total = torch.cat((Y_pred_total, torch.tensor([original_y_pred])), dim=0)

        if print_info is True:
            print(f'Test Index:{test_ind} attack:{att} Original y Pred:{original_y_pred} Attacked y Pred:{adv_y_pred}')
        if save_as_csv is True:
            result = [data.df_name, data.seed, model, eps, test_ind, str(att), str(ground_truth_y.item()),
                      original_y_pred, str(adv_y_pred), attacked_ae, original_ae, max_per, sum_per, cost_time,
                      str(window_range.tolist()).replace(",", ";"), str(X_adv.tolist()).replace(",", ";")]
            append_result_to_csv_file(path_out_dir, result)

    return X_adv_total, Y_adv_total, Y_pred_total


def LBA_attack(data, TSF_model, LBA_model, Y_adv_nvita, sp_location, args, beta, transfer_from='') \
        -> tuple[torch.tensor, torch.tensor, torch.tensor]:
    """
    LBA attack, similar to the Non_LBA_attack.
    For LBA attack, the LBA model is used to find the sensitive points from original adversarial examples
    The original adversarial attack is 1VITA (n=1) in this case, as they produce sparse adversarial attacks
    FGSM and BIM are not suitable for LBA attack.
    Parameter Y_adv_nvita and sp_location are used for the comparison of the original attack.
    If it's a transferability test, Y_adv_nvita and sp_location will be None.
    :param data: the original dataset
    :param TSF_model: the model to be attacked for TSF
    :param LBA_model: the model for learning adversarial patterns from the original adversarial examples
    :param Y_adv_nvita: the attacked prediction by NVITA. None if it's not available
    :param sp_location: a list of the location of the sensitive points. None if it's not available
    :param args: parameters for the attack, using n, epsilon, adv_cnt
    :param beta: the beta for the LBA attack
    :param transfer_from: indicate transferability test if it's not a NULL str,
                          it will only change the file name of the result
    :return: X_adv_total: the adversarial examples,
             Y_adv_total: the attacked prediction,
             Y_pred_total: the original prediction
    """
    path_out_dir = None

    X_adv_total = torch.empty(0).to(Device)
    Y_adv_total = torch.empty(0).to(Device)
    Y_pred_total = torch.empty(0).to(Device)

    # get the parameters from the args
    eps = args.epsilon
    adv_cnt = args.LBA_adv_cnt
    n = args.n
    save_as_csv = args.save_csv
    print_info = args.print_info

    lr = args.LBA_lr
    epochs = args.LBA_epochs
    batch_size = args.LBA_batch_size

    total = data.X_test.shape[0]
    windows_cnt = data.X_test.shape[1]
    features_cnt = data.X_test.shape[2]

    accuracy = 0
    rmse_ori, rmse_lba, rmse_nvita, rmse_nvita_lba = 0.0, 0.0, 0.0, 0.0
    time_cost = 0.0

    if transfer_from != '':
        attack_name = f'df_{data.df_name}_seed_{data.seed}_model_{TSF_model}_epsilon_{eps}_LBA_model_{LBA_model}' \
                      f'_beta_{beta}_lr_{lr}_epochs_{epochs}_batch_{batch_size}_transfer_from_{transfer_from}.csv'
    else:
        attack_name = f'df_{data.df_name}_seed_{data.seed}_model_{TSF_model}_epsilon_{eps}_LBA_model_{LBA_model}' \
                      f'_beta_{beta}_lr_{lr}_epochs_{epochs}_batch_{batch_size}.csv'

    if save_as_csv is True:
        path_out_dir = os.path.join(ATK_PATH, 'LBA', attack_name)
        path_out_dir = check_result_file_path(path_out_dir)
        first_result_line_list = ["df", "Seed", "LBA Model", "Target Model", "n", "Epsilon", "Test Index",
                                  "Attack Name", "True Y", "Original Y Pred", "Attacked Y Pred", "Point", "Feature",
                                  "Window", "Attack Value", "Attacked AE", "Original AE", "Max Per", "Sum Per",
                                  "Cost Time", "Window Range", "Adv Example"]
        create_empty_result_csv_file(path_out_dir, first_result_line_list)

    if print_info:
        print(f'Dataset:{data} LBA Model:{LBA_model} TSF Model:{TSF_model} Epsilon:{eps} Seed:{data.seed} '
              f'Data for LBA training:{adv_cnt} Data to be attacked:{total - adv_cnt}')

    LBA_model.to(Device)
    TSF_model.to(Device)
    LBA_model.eval()
    TSF_model.eval()
    for test_ind in range(adv_cnt, total):
        X_current = data.X_test[test_ind].unsqueeze(0)
        ground_truth_y = data.Y_test[test_ind].unsqueeze(0)
        window_range = data.window_ranges[test_ind]

        # use model to find the sensitive point
        start_time = time.time()
        output_cls, output_atk = LBA_model(X_current)
        cost_time = time.time() - start_time
        _, point = torch.topk(output_cls, n, dim=1)

        perturbations = output_atk.detach()[0].to('cpu').numpy().tolist()
        point = point[0].to('cpu').numpy().tolist()

        ind_class_list = []
        ind_win_list = []
        X_adv = X_current.clone()
        for ind, po in enumerate(point):
            if po == windows_cnt * features_cnt:
                continue
            ind_class, ind_win = po // windows_cnt, po % windows_cnt
            ind_class_list.append(ind_class)
            ind_win_list.append(ind_win)
            X_adv[0][ind_win][ind_class] += beta * perturbations[ind]

        adv_y_pred = TSF_model(X_adv).item()
        original_y_pred = TSF_model(X_current).item()
        original_ae = np.absolute(original_y_pred - ground_truth_y.item())
        attacked_ae = np.absolute(adv_y_pred - ground_truth_y.item())
        eta = X_adv - X_current
        sum_per = torch.sum(torch.abs(eta)).item()
        max_per = torch.max(torch.abs(eta)).item()

        if sp_location is not None and point == sp_location[test_ind]:
            accuracy += 1
        rmse_lba += attacked_ae ** 2
        rmse_ori += original_ae ** 2

        if Y_adv_nvita is not None:
            rmse_nvita += (Y_adv_nvita[test_ind].item() - ground_truth_y) ** 2
            rmse_nvita_lba += (Y_adv_nvita[test_ind].item() - adv_y_pred) ** 2
        time_cost += cost_time

        X_adv_total = torch.cat((X_adv_total, X_adv.clone()), dim=0)
        Y_adv_total = torch.cat((Y_adv_total, torch.tensor([adv_y_pred]).to(Device)), dim=0)
        Y_pred_total = torch.cat((Y_pred_total, torch.tensor([original_y_pred]).to(Device)), dim=0)

        if print_info is True:
            point_real = sp_location[test_ind] if sp_location is not None else 'None'
            print(f'Index:{test_ind} point(real):{point_real} point(prediction):{point} Class:'
                  f'{ind_class_list} Window:{ind_win_list} attack value:{perturbations}')

        if save_as_csv is True:
            result = [data.df_name, data.seed, str(LBA_model), str(TSF_model), n, eps, test_ind,
                      "LBA", str(ground_truth_y.item()), original_y_pred, str(adv_y_pred),
                      str(point), str(ind_class_list), str(ind_win_list), str(perturbations), attacked_ae, original_ae,
                      max_per, sum_per, cost_time, str(window_range.tolist()).replace(",", ";"),
                      str(X_adv.tolist()).replace(",", ";")]
            append_result_to_csv_file(path_out_dir, result)

    return X_adv_total, Y_adv_total, Y_pred_total


def run_LBA_exp(data, TSF_model, args):
    # get the parameters from the args
    n = args.n
    epochs = args.LBA_epochs
    lr = args.LBA_lr
    adv_cnt = args.LBA_adv_cnt
    eps_nvita = args.epsilon
    seed = args.seed
    atk_load_from_csv = args.LBA_ori_atk_load_from_csv
    print_info = args.print_info
    plot_flag = args.plot
    LBA_load_from_pkl = args.LBA_model_load_from_pkl
    LBA_model_save = args.LBA_model_save
    do_transfer_model_test = args.LBA_transfer_atk

    beta_lba_list = args.LBA_beta_list

    ori_attack = f'Non_Targeted_{n}VITA'

    total = data.X_test.shape[0]
    windows_cnt = data.X_test.shape[1]
    features_cnt = data.X_test.shape[2]
    data.X_test = data.X_test.to(Device)
    data.Y_test = data.Y_test.to(Device)

    nVITA_attack_name = f'df_{data.df_name}_seed_{seed}_model_{TSF_model}_epsilon_{eps_nvita}_attack_{ori_attack}.csv'

    LBA_model = CNN_LBA_Model(features_cnt, windows_cnt, n)
    LBA_model_name = 'df_{0}_LBA_Model_{1}_target_model_{2}_seed_{3}_epochs_{4}_lr_{5}_atk_{6}_epsilon_{7}' \
                     '_adv_cnt_{8}_features_{9}_windows_{10}.pt' \
        .format(data.df_name, LBA_model, TSF_model, seed, epochs, lr, ori_attack,
                eps_nvita, adv_cnt, features_cnt, windows_cnt)

    # get the original adversarial examples
    if atk_load_from_csv:
        X_adv, Y_adv, Y_pred = get_Adv_from_csv(os.path.join(ATK_PATH, 'NVITA', nVITA_attack_name))
    else:
        X_adv, Y_adv, Y_pred = Non_LBA_attack(data, TSF_model, args, "NVITA")

    X_adv = X_adv.to(Device)
    Y_adv = Y_adv.to(Device)
    eta = X_adv - data.X_test

    # check the incorrect count of NVITA
    if print_info:
        print("Incorrect count of NVITA: " + str(check_incorrect_cnt_of_NVITA(eta, n)))

    LBA_data = get_LBA_Dataset(data, X_adv, adv_cnt, n, device=Device)
    # get the sensitive point information
    sp_statistics, sp_location = get_total_sensitive_point_from_adv(features_cnt, windows_cnt, data, X_adv, n)
    if print_info:
        print("Sensitive Points Statistics:", sp_statistics)

    # get the LBA model
    if LBA_load_from_pkl is False:
        print('-----begin to train the learning model-----')
        train_model(LBA_data, LBA_model, epochs, lr, print_info=print_info, plot_flag=plot_flag)
        if LBA_model_save:
            torch.save(LBA_model, os.path.join(LBA_MODEL_PATH, LBA_model_name))
    else:
        LBA_model = torch.load(os.path.join(LBA_MODEL_PATH, LBA_model_name))

    for beta in beta_lba_list:
        print("-" * 25 + f'epsilon_lba_atk:{beta}' + "-" * 25)
        X_adv_LBA, Y_adv_LBA, Y_pred = LBA_attack(data, TSF_model, LBA_model, Y_adv, sp_location, args, beta)
        if plot_flag:
            plt.plot(Y_pred.to('cpu'), label='prediction')
            plt.plot(data.Y_test[adv_cnt:total].to('cpu'), label='origin')
            plt.plot(Y_adv_LBA.to('cpu'), label='adv by LBA')
            plt.title(f'Origin model:{TSF_model} LBA model:{LBA_model_name} beta:{beta}')
            plt.legend()
            plt.show()

        if do_transfer_model_test:
            print("\n" + "<" * 30 + "Transferability test" + ">" * 30)
            TSF_model_list = ['CNN', 'LSTM', 'GRU']
            TSF_model_list.remove(str(TSF_model))
            for TSF_transfer_model_name in TSF_model_list:
                TSF_transfer_model = load_model(PATH_ROOT, data.df_name, data.seed, TSF_transfer_model_name).to(Device)

                if print_info:
                    print(f'\n<Transfer Model:{TSF_transfer_model}>\nNo Sensitive Points Statistics')

                _, Y_adv_LBA_tran, Y_pred_new_tran = LBA_attack(data, TSF_transfer_model, LBA_model, None,
                                                                None, args, beta, transfer_from=str(TSF_model))
                if plot_flag:
                    plt.plot(Y_pred_new_tran.to('cpu'), label='prediction')
                    plt.plot(data.Y_test[adv_cnt:total].to('cpu'), label='origin')
                    plt.plot(Y_adv_LBA_tran.to('cpu'), label='atk by LBA')
                    plt.title(f'Transferability test\n'
                              f'Origin model:{TSF_model} Transfer model:{TSF_transfer_model_name}')
                    plt.legend()
                    plt.show()

    return LBA_data, LBA_model, X_adv, Y_adv, Y_pred


# some global variables
PATH_ROOT = os.getcwd()
Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ATK_PATH = os.path.join(PATH_ROOT, 'results')
LBA_MODEL_PATH = os.path.join(PATH_ROOT, 'LBA_models')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, default='Electricity',
                        help='the name of the dataset. options: [CNYExch, Electricity, NZTemp, Oil]')
    parser.add_argument('--model', type=str, required=True, default='CNN',
                        help='the name of the model. options: [CNN, LSTM, GRU]')
    parser.add_argument('--attack', type=str, required=True, default='NVITA',
                        help='the name of the adversarial attack'
                             'options: [NVITA, FGSM, BIM, LBA]'
                             'for LBA, it will set 1VITA as the original attack')
    parser.add_argument('--epsilon', type=float, required=True, help='the epsilon of the original attack')

    # parameters for the LBA attack, using when the attack is LBA
    parser.add_argument('--LBA_transfer_atk', action='store_true', default=False,
                        help='do the transferability test')
    parser.add_argument('--LBA_ori_atk_load_from_csv', action='store_true', default=False,
                        help='as LBA needs to learn the original attack, this flag is used to '
                             'load the original attack result from .csv to accelerate the process, '
                             'instead of running the original attack again.'
                             'be sure the result file is existed.')
    parser.add_argument('--LBA_model_save', action='store_true', default=False, help='save the LBA model')
    parser.add_argument('--LBA_model_load_from_pkl', action='store_true', default=False,
                        help='load the LBA model from .pkl file, instead of training the model again')
    parser.add_argument('--LBA_beta_list', type=list, default=[0.75, 1.0, 1.5, 1.75],
                        help='the beta list for the LBA attack')
    parser.add_argument('--LBA_adv_cnt', type=int, default=100,
                        help='the count of the adversarial examples as the training data'
                             'can not larger than the total test data count(250 as Default)')
    parser.add_argument('--LBA_epochs', type=int, default=50, help='the epochs for the LBA model training')
    parser.add_argument('--LBA_batch_size', type=int, default=25, help='the batch size for the LBA model training')
    parser.add_argument('--LBA_lr', type=float, default=0.001, help='the learning rate for the LBA model')

    # parameters for the Non-Targeted NVITA
    parser.add_argument('--n', type=int, default=1, help='the n value for the NVITA attack.'
                                                         'the implementation of NVITA is always non-targeted')
    parser.add_argument('--maxiter', type=int, default=60, help='the maxiter for the NVITA attack')
    parser.add_argument('--tol', type=float, default=0.01, help='the tolerance for the NVITA attack')

    # other parameters
    parser.add_argument('--print_info', action='store_true', default=False, help='print the information in process')
    parser.add_argument('--plot', action='store_true', default=False, help='plot the result')
    parser.add_argument('--save_csv', action='store_true', default=False, help='save the attack result as .csv')
    parser.add_argument('--seed', type=int, default=2210, help='the seed for the random number generator')
    parser.add_argument('--cpu', action='store_true', default=False, help='use cpu to run the code')

    args = parser.parse_args()

    set_seed(args.seed)
    if args.cpu:
        Device = torch.device("cpu")

    print('Args:', args)
    print('Time:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print('ROOT:', PATH_ROOT)
    print('Device:', Device)

    # windows_size: 4--Electricity 3--NZTemp 7--CNYExch 7--Oil
    data = get_ori_data(args.dataset, PATH_ROOT, args.seed)
    TSF_model = load_model(PATH_ROOT, args.dataset, args.seed, args.model)

    if args.LBA_adv_cnt >= data.X_test.shape[0]:
        raise ValueError('The count of the adversarial examples for LBA '
                         'should not be larger than the total test data count')

    if args.attack == 'LBA':
        LBA_data, LBA_model, X_adv, Y_adv, Y_pred = run_LBA_exp(data, TSF_model, args)
    else:
        X_adv_total, Y_adv_total, Y_pred_total = Non_LBA_attack(data, TSF_model, args, attack=args.attack)
