import sys, os
from mpi4py import MPI
import numpy as np

sys.path.append("..")
from general_utils.utils import make_intervals, print_wise

def parallel_setup():
    comm = MPI.COMM_WORLD  # Get the global communicator
    rank = comm.Get_rank()  # Get the rank (ID) of the current process
    size = comm.Get_size()  # Get the total number of processes
    return comm, rank, size


def split_parallel(n_tasks, func_work, args_work, paths, rec_back=False, func_merge=None, args_merge=None):
    comm, rank, size = parallel_setup()
    root = 0
    tasks = make_intervals(n_tasks, size-1)
    if rank == root:
        print("tasks: ")
        print_wise(tasks)
    # end if rank == root:

    next_to_do = 0
    if rank == root:
        for dst in range(1, size):
            print_wise(dst)
            comm.send(
                np.int32(next_to_do), dest=dst, tag=11
            )  # starts sending data to process with rank 1
            next_to_do += 1
            print_wise(f"sent {next_to_do}", rank=rank)
        # end for dst in range(1, size):

        if rec_back:
            func_merge(comm, size, rank, paths, *args_merge)
        # end if rec_back:
    
    else:
        data = comm.recv(source=root, tag=11)  # Receive data from process with rank 0
        task_to_do = tasks[data]
        print_wise(f"received: task {task_to_do}", rank=rank)
        func_work(*task_to_do, comm, rank, root, paths, *args_work)
    # end if rank == root:

    print_wise("finished", rank=rank)
    MPI.Finalize()
# EOF


def master_workers_queue(task_list, paths, func, *args, **kwargs):
    comm, rank, size = parallel_setup()
    root = 0
    tot_n = len(task_list)
    next_to_do = 0
    if rank == 0:
        for dst in range(1, size):
            comm.send(
                np.int32(next_to_do), dest=dst, tag=11
            )  # Send data to process with rank 1
            next_to_do += 1
            print_wise(f"sent {next_to_do} to {dst}", rank=rank)
            if next_to_do == tot_n:
                break
            # end if done_by_now+1 > tot_n:
        # end for dst in range(1, size):
        
        while next_to_do < tot_n:
            status = MPI.Status()
            d = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            src = status.Get_source()
            tag = status.Get_tag()
            comm.send(
                np.int32(next_to_do), dest=src, tag=11
            )  # Send data to process with rank 1
            next_to_do += 1
            print_wise(f"{src} is free again, root sent : {next_to_do}", rank=rank)
        # end while next_to_do < tot_n:

        print_wise(f"Sending termination signals", rank=rank)
        for i in range(1, size):
            comm.send(np.int32(-1), dest=i, tag=11)  # Send data to process with rank 1
        # end for i in range(1, size):

    else:
        while True:
            data = comm.recv(source=0, tag=11)  # Receive data from process with rank 0
            print_wise(f"received: {data}", rank=rank)
            if data == np.int32(-1):
                break
            func(paths, rank, task_list[data], *args, **kwargs)
            comm.send(
                np.int32(1), dest=root, tag=11
            )  # Send data to process with rank 1
        # end while True:
    # end if rank == 0:

    print_wise("finished", rank=rank)
    MPI.Finalize()
#EOF



def master_merger_queue(task_list, paths, func, *args, **kwargs):
    comm, rank, size = parallel_setup()
    root = 0
    merger = 1
    tot_n = len(task_list)
    next_to_do = 0
    if rank == 0:
        for dst in range(2, size):
            print_wise(f"sending stuff", rank=rank)
            comm.send(
                np.int32(next_to_do), dest=dst, tag=11
            )  # Send data to process with rank 1
            next_to_do += 1
            print_wise(f"computed {next_to_do}", rank=rank)
            if next_to_do == tot_n:
                break
            # end if done_by_now+1 > tot_n:

        # spotlight = np.zeros(size - 1)  # one means the process is free
        while next_to_do < tot_n:
            status = MPI.Status()
            d = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            src = status.Get_source()
            tag = status.Get_tag()
            comm.send(
                np.int32(next_to_do), dest=src, tag=11
            )  # Send data to process with rank 1
            next_to_do += 1
            print_wise(f"received from {src} , root : {next_to_do}", rank=rank)
        for i in range(2, size):
            comm.send(np.int32(-1), dest=i, tag=11)  # Send data to process with rank 1

    elif rank == 1: # HERE TO CHANGE STUFF
        model_names = args[0]
        paths = args[7]
        n_batches = args[4]
        gram_or_cov = args[5]
        w, h = len(get_relevant_output_layers(model_names[0])), len(
            get_relevant_output_layers(model_names[1])
        )  # n layers of model_names[0] and model_names[1]
        cka_mat = np.zeros((w, h))

        counter = 0
        while counter < tot_n:
            status = MPI.Status()
            d = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            counter += 1
            print_wise(f"received {d}, {counter} tasks already processed", rank=rank)
            cka_mat[d[0], d[1]] = d[2]
            print(cka_mat)
        csv_save_path = f"{paths['results_path']}/cka_{model_names[0]}_{model_names[1]}_{n_batches}_batches_{gram_or_cov}.csv"
        np.savetxt(csv_save_path, cka_mat, delimiter=",")

    else:
        model_names = args[0]
        while True:
            data = comm.recv(source=0, tag=11)  # Receive data from process with rank 0
            print_wise(f"received: {data}", rank=rank)
            if data == np.int32(-1):
                break
            print_wise(f"starting cka...", rank=rank)
            res = func(rank, task_list[data], *args)
            to_send = perm2idx(rank, task_list[data], model_names)
            to_send.append(res)
            comm.send(to_send, dest=merger, tag=11)  # Send data to process with rank 1
            comm.send(
                np.int32(1), dest=root, tag=11
            )  # Send data to process with rank 1
            print_wise(f"free again", rank=rank)

    print_wise("finished", rank=rank)
    MPI.Finalize()

