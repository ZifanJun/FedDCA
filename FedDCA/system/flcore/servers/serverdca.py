import time
from system.flcore.clients.clientDCA import clientDCA
from system.flcore.servers.serverbase import Server
from threading import Thread
from concurrent.futures import ThreadPoolExecutor


class FedDCA(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientDCA)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    def train(self):
        for i in range(self.global_rounds+1):

            self.selected_clients = self.select_clients()
            self.send_models()
            s_t = time.time()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            # 阶段训练
            print("Starting stage training...")
            threads = [Thread(target=client.train) for client in self.selected_clients]
            [t.start() for t in threads]
            [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

    # def send_models(self):
    #     assert (len(self.clients) > 0)
    #
    #     for client in self.clients:
    #         client.set_parameters(self.global_model)
    #         client.adaptive_aggregate()

    def send_models(self):
        def send_model_to_client(client):
            client.set_parameters(self.global_model)
            client.adaptive_aggregate()

        threads = [Thread(target=send_model_to_client, args=(client,)) for client in self.clients]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
