import numpy as np
import csv
import random
import math
import matplotlib.pyplot as plt

# csv파일 이용여부를 True & False로 지정한다.
# csv파일은 아래와 같은 포맷으로 저장되어야한다.
'''
NodeName_1, x_1, y_1
NodeName_2, x_2, y_2
NodeName_3, x_3, y_3
    .........
    .........
NodeName_n, x_n, y_n
'''
csv_cities = True
csv_name = 'C:/python/meta_Heuristics/ALL_tsp/bayg29.csv'


# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# # # # Simulated Annealing for TSP
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================


list_of_Nodes = []


class Node:
    
    def __init__(self, name, x, y):
        
        self.name = name
        self.x = x
        self.y = y
        self.L2 = {}
        self.total_candidate = {}
        
        list_of_Nodes.append(self)
        
        
    def calculate_L2_distacne(self):        
        for node in list_of_Nodes:
            L2_norm = np.linalg.norm((self.x-node.x, self.y-node.y))
            self.L2[node.name] = L2_norm
        del self.L2[self.name]
        
    def set_total_candidata_list(self):
        for node in list_of_Nodes: 
            self.total_candidate[node.name] = True
        del self.total_candidate[self.name]





class SA:
    
    def __init__(self, 
                 using_heuristics = False,
                 temperature = 500,
                 alpha = 0.99,
                 epochs = 1000):
        
        
        #적응적 내부루프를 수행을 위한 new_length저장용 리스트
        self.new_length_history = []
        self.new_value_history = []
        
        
        #계속 갱신해나갈 베스트 솔루션값의 초기값 설정.
        self.best_solution = np.inf
        
        #노드
        self.set_node()           
        
        self.length = 0.0
        self.total_epochs = epochs
        self.inside_epochs = epochs
        self.temperature = temperature
        self.alpha = alpha
        self.using_heuristics = using_heuristics
        
     
      
        self.Initial_value_setting(using_heuristics = self.using_heuristics)
        self.cooling_scheduling()
        
        
        
        
    def set_node(self):
        if csv_cities:
            self.set_node_using_csv()
        else:
            self.set_node_inside()
      
        
    def set_node_inside(self):
        
        # 이런식으로 직접 도시를 입력해주거나
        # csv파일을 사용한다.
        n1 = Node('c1',64,96)     
        n2 = Node('c2',80,39)    
        n3 = Node('c3',69,23)    
        n4 = Node('c4',72,42) 
        n5 = Node('c5',48,67) 
        n6 = Node('c6',58,43) 
        n7 = Node('c7',81,34) 
        n8 = Node('c8',79,17) 
        n9 = Node('c9',30,23) 

        for node in list_of_Nodes:
            node.calculate_L2_distacne()
            node.set_total_candidata_list()


    def set_node_using_csv(self):    
       
        with open(csv_name, 'rt') as f:
            reader = csv.reader(f)
           
            for row in reader:
                
                Node(row[0],float(row[1]),float(row[2]))
        
                for node in list_of_Nodes:
                    node.calculate_L2_distacne()
                    node.set_total_candidata_list()
        
        
        
        
        
    def Initial_value_setting(self, using_heuristics = None):
        # SA의 초기해는 무작위 or 휴리스틱에 의하여 결정하며
        # using_heuristics여부에 따라 초기해 설정을 달리한다.
        
        # 휴리스틱에 의한 초기해 설정은 초기부터 좋은해로 시작한다는 장점이 있으나
        # 선택된 휴리스틱방법에 따라 그 결과가 달라지며 최적해 수렴시간단축을 보장하진 못한다.
        # 또한 휴리스틱방법으로 설정된 초기해가 지역최적점에 가까운 경우엔
        # 이로 인하여 그 지역점을 벗어나기 어려운 경우가 생길 수도 있으니 주의해야한다.
        # 해당 코드에서는 휴리스틱법으로 Lnn법을 사용하였으며
        # 이는 ACO법의 초기 페로몬 업데이트에서 tau0를 얻기위해 사용되는 방법론과 같다.
        
        # 또한  Lnn법으로 초기값을 설정한다 하더라도
        # 이웃구조를 랜덤 하위경로역순으로 하고 있는 해당 프로그램에서는 탁월한 성능을 보이기엔 어렵다.
        
        
        if using_heuristics == False:
            # list_of_Nodes의 순서를 무작위로 바꾸어 초기해로 만들어준다.
            self.init_value = sorted(list_of_Nodes, key=lambda *args: np.random.rand())
            self.value = self.init_value
            
            # 무작위로 바뀐 노드순서의 총 길이를 구한다.
            # 즉, 초기해의 품질을 계산한다.
            for node in self.init_value:
                next_node = self.init_value[self.init_value.index(node)-len(self.init_value)+1]
                self.length += node.L2[next_node.name]
            self.init_length = self.length
                
                
        else:
            #휴리스틱해인 Lnn을 구한다.
            Lnn = []
            self.Lnn_length = []
            self.init_value = []
    
    
            #출발도시를 선정하기위한 코드
            #도시의 누적확률값 - (0~1)난수값에서 첫번째로 양수가 출력되는 index가 출발도시가 된다.
            1/len(list_of_Nodes)
    
            first_node = []
            for i in range(len(list_of_Nodes)):
                first_node.append(1/len(list_of_Nodes)*(i+1))
    
            TF_list = list( (np.array(first_node) - np.random.rand())>0 )

    
            #index는 다음도시를 찾을때 list_of_cities리스트의 인덱스탐색용
            #name은 딕셔너리에서 해당 도시 탐색용으로 저장,
            #그리고 total_candidta를 다시 계산하므로 del을 통하여 자기를 삭제해주는것 중요함.
            #또한 Lnn에 도시이름을 저장해준다.
            next_node_index = TF_list.index(True)
            next_node = list_of_Nodes[TF_list.index(True)].name
            Lnn.append(next_node)
            self.init_value.append(list_of_Nodes[next_node_index])
    
            #모든 도시객체의 total_candidate딕셔너리에서
            #시작도시로 선택된 도시를 True에서 False로 바꿔주는 작업을 한다.
            for node in list_of_Nodes:
                node.total_candidate[next_node] = False
            del list_of_Nodes[next_node_index].total_candidate[next_node]
            

    

            # =============================================================================
            # #Lnn 계산
            # =============================================================================
            for banbok in range(len(list_of_Nodes)-1):
    
                distance_to_value = np.array(list(list_of_Nodes[next_node_index].L2.values()))
                total_candidate_value = np.array(list(list_of_Nodes[next_node_index].total_candidate.values()))
                
                # total * distance로 이루어져있고, total엔 False값이 포함되있기때문에
                # 해당되는 distance는 0으로 처리된다.
                isEmpty_CL_value = total_candidate_value*distance_to_value
    
                # 다음도시에 도착.
                # isEmpty_CL_value에서 0으로 처리되지 않은 애들만 가지고 min을 구하고싶으므로
                # np.nonzero(isEmpty_CL_value)를 사용한다.
                # np.nonzero는 0이 아닌 index들을 리턴해준다.
                for i, item in enumerate(list_of_Nodes[next_node_index].L2.items()):
                    if item[1]*total_candidate_value[i] == min(isEmpty_CL_value[np.nonzero(isEmpty_CL_value)]):
                        next_node = item[0]
    
                for i ,node in enumerate(list_of_Nodes):
                    if node.name == next_node:
                        next_node_index = i
            
                Lnn.append(next_node)
                self.Lnn_length.append(min(isEmpty_CL_value[np.nonzero(isEmpty_CL_value)]))
                self.init_value.append(list_of_Nodes[next_node_index])
                
                for node in list_of_Nodes:
                    node.total_candidate[next_node] = False
                del list_of_Nodes[next_node_index].total_candidate[next_node]


            # 마지막도시와 첫번째도시를 이어줌으로써 최종 거리를 구한다.
            self.Lnn_length.append(list_of_Nodes[next_node_index].L2[Lnn[0]])
            self.value = self.init_value
            self.length = sum(self.Lnn_length)
            self.init_length = self.length
        
    def calculate_new_value(self):
        
        # 새로운 이웃해를 생성하기 위한 방법으로
        # 하위경로역순 방법을 이용한다.

        # start_pos의 range()를 1부터 두는 이유는
        # start와 end가 첫번째, 마지막노드가 선정되는 경우, 아무런 변화가 없기때문에
        # 이를 방지하기위함이다.
        
        
        self.new_length = 0
        
        start_pos = random.randint(1,len(list_of_Nodes)-2)
        end_pos = random.randint(start_pos+1,len(list_of_Nodes)-1)    
        
        self.new_value = [None for i in range(len(list_of_Nodes))]
        
        # for문에 range(0,1)이면 1까지가 아니라 0까지만 돌아가므로
        # end_pos+1을 해준다.
        # 아래의 for문을 통해 하위경로역순으로 이웃해를 구하며
        
        # 만약 기존의 해가 123456789이었다면
        # 하위경로역순의 해는 ****765****의 결과가 나온다.
        for i, x in enumerate(range(start_pos, end_pos+1)):
            self.new_value[x] = self.value[end_pos-i]

        # 이제 ****765****에서 *부분을 채위주는 for문을 사용한다.
        # 아래의 코드를 통해 123476589가 된다.
        for i in range(len(list_of_Nodes)):
            if self.new_value[i]==None:
                self.new_value[i] = self.value[i]

        #새롭게 구해진 해의 length를 계산한다.                
        for node in self.new_value:
            next_node = self.new_value[self.new_value.index(node)-len(self.new_value)+1]
            self.new_length += node.L2[next_node.name] 
        
        self.new_length_history.append(self.new_length)
        self.new_value_history.append(self.new_value)
            
            
    def Acceptance_criterion(self):
        
        #새로운 해 계산
        self.calculate_new_value()
        
        if self.new_length < self.length:
            self.length = self.new_length
            self.value = self.new_value
        
        elif np.exp(-(self.new_length-self.length)/self.temperature) > np.random.rand():
            self.length = self.new_length
            self.value = self.new_value
            
            
    def cooling_scheduling(self):
        
        '''        
        SA에서는 내부루프의 반복수보다 온도의 하락폭을 줄이는 것이
        해의 성능측면에서 유리하다고 알려져 있다.(Johnson et al., 1989)
        
        본 모형에서느 내부루프 반복수를 상황에 따라 변화하게하는 적응적 방법론을 사용하였으며
        적응적 방법론 중 (Ali et al., 2002)를 참고하였다.
        
        온도 강하 방법으로는 기하 스케줄링 방법을 사용하였다.
        
        '''
        
        ### 멈춤규칙 : 온도 0.0001이하 ###
        self.change_of_temperature = []
        self.change_of_temperature.append(self.temperature)
        
        self.history_of_bestLength = []
        self.history_of_bestLength.append(self.init_length)
        
        while self.temperature > 0.0001:
            
            for banbok in range(int(self.inside_epochs)):
                self.Acceptance_criterion()


            #cooling_scheduling방법론 중에서,  Geometric_scheduling사용
            self.temperature = self.alpha*self.temperature
                
            #적응적 내루루프사용을 위한 f_H, f_L
            f_H = max(self.new_length_history)
            f_L = min(self.new_length_history)
            F = 1-np.exp(-(f_H-f_L)/f_H)

            inside_epoch_scheduling = math.floor(self.total_epochs*F)
            self.inside_epochs = self.total_epochs + inside_epoch_scheduling
            
            
            # 현재까지 발견한 해 중에서 가장 좋은 해를 저장한다.
            if f_L < self.best_solution:
                self.best_solution = f_L
                self.best_value_index = self.new_length_history.index(min(self.new_length_history))
                self.best_value = self.new_value_history[self.best_value_index]
            
            #적응적 내부루프를 epoch마다 사용하기위하여 바깥 for문마다 해당 리스트를 초기화
            self.new_length_history = []
            self.new_value_history = []
            self.change_of_temperature.append(self.temperature)
            self.history_of_bestLength.append(self.best_solution)
            
            #print('\nepochs :',epoch+1)
            print('\ntemperature :',self.temperature)
            print('inside_epochs :',self.inside_epochs)
            print('current_distance :',self.length)
            print('best_solution :',self.best_solution)
            
           
            
        ### 최종결과 print ###  
        self.final_route = []
        self.best_x=[]
        self.best_y=[]
        
        for node in self.best_value:
            next_node = self.best_value[self.best_value.index(node)-len(self.best_value)+1]
            self.final_route.append(next_node.name)
            self.best_x.append(next_node.x)
            self.best_y.append(next_node.y)
        self.best_x.append(self.best_value[1].x)
        self.best_y.append(self.best_value[1].y)

        print("\n\n###############")            
        print("### Result! ###")
        print("###############\n")
        print("best route:",self.final_route)
        print("Initial best distance:",self.init_length)
        print("final distacne:", self.best_solution)
        
        
        #시각화
        plt.figure(figsize=(14,28))
        
        #시각화1. best_distacne값의 변화
        plt.subplot(2,1,1)
        plt.title('Change of Best Distance according to Temperature')
        plt.xlabel('Temperature')
        plt.ylabel('Best Distance')
        plt.plot(self.change_of_temperature, self.history_of_bestLength, c='r', label="best Distance" )
        plt.gca().invert_xaxis() #X축 반전을 위한 코드 
        plt.legend(loc='upper right')
        plt.grid(True)
        
        #시각화2. 최종 best_route
        plt.subplot(2,1,2)
        plt.title('Best Route')
        for i in range(len(list_of_Nodes)):
            plt.text(self.best_x[i]-np.mean(self.best_x)/40, self.best_y[i]+np.mean(self.best_y)/40, "{}".format(self.final_route[i]), fontsize=10)    
        plt.plot(self.best_x,self.best_y,c="b",
                 lw=2, ls="-", marker="o", ms=10, mec="black", mew=1, mfc="white")        

            
# =============================================================================
# application
# =============================================================================    
SA(using_heuristics = False,
   temperature = 500,
   alpha = 0.99,
   epochs = 1000)



