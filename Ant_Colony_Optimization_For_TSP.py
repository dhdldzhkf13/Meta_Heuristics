import csv
import numpy as np
import operator
import matplotlib.pyplot as plt


list_of_cities =[]

# csv파일 이용여부
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
csv_name = 'C:/python/meta_Heuristics/ALL_tsp/st70.csv'


#class___1, 개미 군체 최적화법을 사용하기위한 infra
class City:

    
    # 해당 City클래스의 생성자로 이 생성자는 name, x, y, distance_to의 4가지 매개변수로 이루어짐.
    def __init__(self, name, x, y, distance_to=None): 
        
        
        self.name = name        
        self.x = x
        self.y = y
        
        list_of_cities.append(self) 
        
        # 각 도시간의 거리
        self.distance_to = {self.name: 0}
        if distance_to:
            self.distance_to = distance_to
            
            
        # 페로몬을 만들어준다. 
        self.pheromone_to = {self.name: 0}
        self.candidate = {self.name: 0}
        self.total_candidate = {self.name: 0}
        
        
        
        


    # 두 도시의 유클리디안 거리를 구하기위한 메소드를 만듦
    def point_dist(self, x1,y1,x2,y2):
        return ((x1-x2)**2 + (y1-y2)**2)**(0.5)

    # 거리계산
    def calculate_distances(self): 
        for city in list_of_cities: # list_of_cities == 클래스 City의 객체들을 모아놓은 리스트
            
            #위에서 짠 point_dist를 사용하여 각 도시마다의 거리를 계산한다.
            #key값에는 도시이름이, value값에는 거리가 저장된다.
            tmp_dist = self.point_dist(self.x, self.y, city.x, city.y)
            self.distance_to[city.name] = tmp_dist
        del self.distance_to[self.name]
            
                     
    # 주요 후보집합(Tabu Search이론)
    def set_candidata_List(self, number_of_CL = None):
        
        #딕셔너리형을 내림차순으로 sorted하기위해서 아래의 코드를 이용한다.
        #리턴되는 자료형은 딕셔너리형이 아닌 arg값이 각각의 tuple인 list형이다.
        candidate = sorted(self.distance_to.items(), key = operator.itemgetter(1), reverse=0)[0:number_of_CL]  
        for i in candidate:
            #아직 모든 후보지역들은 선택되기 전이므로 참값을준다.
            #만약 선택되었다면 그때마다 거짓값으로 바꿔줘야할것이다.
            self.candidate[ i[0] ] = True
        del self.candidate[self.name]
        
        
    # 전체 후보집합
    def set_total_candidata_list(self):
        for city in list_of_cities: # list_of_cities == 클래스 City의 객체들을 모아놓은 리스트
            self.total_candidate[city.name] = True
        del self.total_candidate[self.name]
        
        
        
    # 페로몬 생성
    def initial_pheromone(self): 
        for city in list_of_cities: # list_of_cities == 클래스 City의 객체들을 모아놓은 리스트
            self.pheromone_to[city.name] = 0
        del self.pheromone_to[self.name]
   

    
    
    
     

# 실질적인 개미 군체 최적화법을 구현하는 클래스
class ACO:

    def __init__(self, 
                 number_of_CL = 10,
                 num_of_ants = None,
                 alpha = 1,
                 beta = 2,
                 total_epochs = 1000,
                 rho = 0.1,
                 q0 = 0.9): 
        
        
        #set.city에서 number_of_CL 필요하기때문에
        #number_of_CL필드는 먼저 세팅.
        self.number_of_CL = number_of_CL



        # =============================================================================         
        self.set_city()
        # =============================================================================


        #아래의 파라미터들은 list_of_cities리스트가 채워져야 실행가능한 코드이기 때문에
        #set.city아래에 적어준다.
        if num_of_ants == None:
            self.num_of_ants = len(list_of_cities)
        else:
            self.num_of_ants = num_of_ants
        self.alpha = alpha
        self.beta = beta
        self.total_epochs = total_epochs
        self.rho = rho
        self.q0 = q0

        #도시와 파라미터들의 세팅이 끝났으므로, ACO를 실행한다.
        self.update_initial_pheromone()
        self.update_pheromone()
        
        
        
    def set_city(self):
        if csv_cities:
            self.set_city_using_csv()
        else:
            self.set_city_inside()
      
        
    def set_city_inside(self):
        
        # 이런식으로 직접 도시를 입력해주거나
        # csv파일을 사용한다.
        c1 = City('c1',64,96)     
        c2 = City('c2',80,39)    
        c3 = City('c3',69,23)    
        c4 = City('c4',72,42) 
        c5 = City('c5',48,67) 
        c6 = City('c6',58,43) 
        c7 = City('c7',81,34) 
        c8 = City('c8',79,17) 
        c9 = City('c9',30,23) 

        for city in list_of_cities:
            
            #위에서 구현한 City클래스를 이용하여 infra를 갖춘다.
            City.calculate_distances(city) 
            City.set_candidata_List(city, self.number_of_CL)
            City.set_total_candidata_list(city)
            City.initial_pheromone(city)



    def set_city_using_csv(self):    
       
        with open(csv_name, 'rt') as f:
            reader = csv.reader(f)
           
            for row in reader:
                City(row[0],float(row[1]),float(row[2]))
           
            for city in list_of_cities:
                #위에서 구현한 City클래스를 이용하여 infra를 갖춘다.
                City.calculate_distances(city) 
                City.set_candidata_List(city, self.number_of_CL)
                City.set_total_candidata_list(city)
                City.initial_pheromone(city)
                print('node_setting... :',city.name)
                       
        
        
    def update_initial_pheromone(self):
        
        #초기 페로몬을 마련하기 위해서
        #휴리스틱해인 Lnn을 구한다.
        Lnn = []
        Lnn_length = []


        #출발도시를 선정하기위한 코드
        #도시의 누적확률값 - (0~1)난수값에서 첫번째로 양수가 출력되는 index가 출발도시가 된다.
        1/len(list_of_cities)

        first_city = []
        for i in range(len(list_of_cities)):
            first_city.append(1/len(list_of_cities)*(i+1))

        TF_list = list( (np.array(first_city) - np.random.rand())>0 )

        #list의 index함수는 True값이 여러개임에도 불구하고 첫번째로 발견되는 Ture값의 index만 리턴해준다.
        TF_list.index(True)


        #index는 다음도시를 찾을때 list_of_cities리스트의 인덱스탐색용
        #name은 딕셔너리에서 해당 도시 탐색용으로 저장,
        #그리고 total_candidta를 다시 계산하므로 del을 통하여 자기를 삭제해주는것 중요함.
        #또한 Lnn에 도시이름을 저장해준다.
        next_city_index = TF_list.index(True)
        next_city = list_of_cities[TF_list.index(True)].name
        Lnn.append(next_city)

        #모든 도시객체의 total_candidate딕셔너리에서
        #시작도시로 선택된 도시를 True에서 False로 바꿔주는 작업을 한다.
        #여기서는 total_candidate에만 적용하지만
        #로컬업데이트할때는 candidate_list에도 적용해야한다.
        for city in list_of_cities:
            city.total_candidate[next_city] = False
        del list_of_cities[next_city_index].total_candidate[next_city]


        # =============================================================================
        # #Lnn 계산
        # =============================================================================

        for banbok in range(len(list_of_cities)-1):

            distance_to_value = np.array(list(list_of_cities[next_city_index].distance_to.values()))
            total_candidate_value = np.array(list(list_of_cities[next_city_index].total_candidate.values()))
            
            # total * distance로 이루어져있고, total엔 False값이 포함되있기때문에
            # 해당되는 distance는 0으로 처리된다.
            isEmpty_CL_value = total_candidate_value*distance_to_value

            # 다음도시에 도착.
            # isEmpty_CL_value에서 0으로 처리되지 않은 애들만 가지고 min을 구하고싶으므로
            # np.nonzero(isEmpty_CL_value)를 사용한다.
            # np.nonzero는 0이 아닌 index들을 리턴해준다.
            for i, items in enumerate(list_of_cities[next_city_index].distance_to.items()):
                if items[1]*total_candidate_value[i] == min(isEmpty_CL_value[np.nonzero(isEmpty_CL_value)]):
                    next_city = items[0]

            for i ,city in enumerate(list_of_cities):
                if city.name == next_city:
                    next_city_index = i
        
            Lnn.append(next_city)
            Lnn_length.append(min(isEmpty_CL_value[np.nonzero(isEmpty_CL_value)]))
        
            for city in list_of_cities:
                city.total_candidate[next_city] = False
            del list_of_cities[next_city_index].total_candidate[next_city]


        # 마지막도시와 첫번째도시를 이어줌으로써 최종 거리를 구한다.
        # Lnn을 이용하여 tau0를 계산
        Lnn_length.append(list_of_cities[next_city_index].distance_to[Lnn[0]])
        self.tau0 = 1/(len(list_of_cities)*sum(Lnn_length))
        
        #최종적으로 tau0를 dictionary에 넣어준다.
        #추가적으로 total_candidate를 다시 True로 바꿔줘야한다.
        for pivot_city in list_of_cities:
            for subordinate_city in list_of_cities:
                pivot_city.pheromone_to[subordinate_city.name] = self.tau0
                pivot_city.total_candidate[subordinate_city.name] = True
            del pivot_city.pheromone_to[pivot_city.name]
            del pivot_city.total_candidate[pivot_city.name]
            
            
            
    def update_pheromone(self):
        # =============================================================================
        # =============================================================================
        # =============================================================================
        # # # local_pheromone update 
        # =============================================================================
        # =============================================================================
        # =============================================================================   
        
        self.history_of_bestLength = []
        best_length = np.inf
        for epoch in range(self.total_epochs):  
          
            history_ant = []
            history_ant_index = []
            history_length = []
            
        
            for local_iterations in range(self.num_of_ants):
                
                ant = []
                ant_index = []
                ant_length = []
                
                
                #출발도시를 선정하기위한 코드
                #도시의 누적확률값 - (0~1)난수값에서 첫번째로 양수가 출력되는 index가 출발도시가 된다.                
                first_city = []
                for i in range(len(list_of_cities)):
                    first_city.append(1/len(list_of_cities)*(i+1))
                
                TF_list = list( (np.array(first_city) - np.random.rand())>0 )
                
                
                #index는 다음도시를 찾을때 list_of_cities리스트의 인덱스탐색용
                #name은 딕셔너리에서 해당 도시 탐색용으로 저장,
                next_city_index = TF_list.index(True)
                next_city = list_of_cities[next_city_index].name
                ant.append(next_city)
                ant_index.append(next_city_index)
                
                #모든 도시객체의 total_candidate딕셔너리에서
                #시작도시로 선택된 도시를 False에서 True로 바꿔주는 작업을 한다.
                for city in list_of_cities:
                    city.total_candidate[next_city] = False
                del list_of_cities[next_city_index].total_candidate[next_city]
                
                for city in list_of_cities:
                    if next_city in city.candidate:
                        city.candidate[next_city] = False
                        
                

                # =============================================================================
                # =============================================================================
                # # # # 여기까지 첫번째도시를 결정해주는 코드였음, 이제 ACO의 규칙에 따라서 다음도시를 선택해줘야한다.
                # =============================================================================
                # =============================================================================

                #step1. q0=0.9미만이면 최적경로로, 아니면 확률값에따라 다음 도시를 결정하게함.
                #보통 alpha=1, beta = 2~5, q0 = 0.9로 둔다.
                #alpha가 0에 가까울수록 가까운 도시가 선택될 확률이 커지고
                #beta가 0에 가까울수록 지역최적해로 수렴할 확률이 커진다.
                #q0가 1에 가까울수록 지역최적해만 선택하게됨.
                        
                #근데 여기서 candidate먼저 고려해야하므로
                #candidtae딕셔너리의 key값을 이용하여 candidate들의 페로몬에만 접근해야한다.
             
                
                # =============================================================================
                # for문을 통하여 개미 한마리가 완주하게끔 도와줌.
                # =============================================================================
                
                for banbok in range(len(list_of_cities)-1):
                
                    #candidate들의 key값
                    candidate_keys = list(list_of_cities[next_city_index].candidate.keys())
                
                    candidate_pheromone_to = {}
                    candidate_distace_to = {}
                
                    for key in candidate_keys:
                        candidate_pheromone_to[key] = list_of_cities[next_city_index].pheromone_to[key]
                        candidate_distace_to[key] = list_of_cities[next_city_index].distance_to[key]
                
                
                    #isEmpty가 False라면 무조건 0이되므로 max값에서 자연스럽게 걸러지게 된다.
                    pheromone = list((candidate_pheromone_to.values()))
                    distance = np.array(list((candidate_distace_to.values()))) 
                    isEmpty = list(list_of_cities[next_city_index].candidate.values())
                
                
                    #우선 첫번째 if문을 통해서 candidate들 중에서 방문 안한곳이 있는지 찾는다.
                    nansoo = np.random.rand()
                    if sum(isEmpty)==0:
                        nearest_neighbor_city = np.array(list(list_of_cities[next_city_index].total_candidate.values()))
                        nearest_neighbor_distance = np.array(list(list_of_cities[next_city_index].distance_to.values()))
                
                        nearest_neighbor = nearest_neighbor_city*nearest_neighbor_distance
                        
                        nearest_neighbor[np.nonzero(nearest_neighbor)]
                
                        #item[0] = keys
                        #item[1] = values
                        for i, items in enumerate(list_of_cities[next_city_index].distance_to.items()):
                            if items[1]*nearest_neighbor_city[i] == min(nearest_neighbor[np.nonzero(nearest_neighbor)]):
                                next_city = items[0]
                
                        for i ,city in enumerate(list_of_cities):
                            if city.name == next_city:
                                next_city_index = i
                
                    elif nansoo <= self.q0:
                        #rand<q0인 경우
                        
                        # candidate들 중에서 가장 좋은 루트를 찾기위한 코드
                        # Tau * Eta * isEmpty
                        J_list = ((pheromone)*( (1/distance)**self.beta )*(isEmpty)) 
                
                        # max값을 통해서 해당 루트를 찾는다.
                        # 그리고 max(J_list)가 N개의 candidate리스트의 어느 인덱스에 있는지를 이용하여서
                        # max(J_list)가 어느 도시인지 정확히 알아낸다.
                        # J_list와 위의 distacne의 순서는 같고 distance의 순서는 candidate_distace_to 순서와 같으므로
                        # distance를 통해서 다음 도시의 정확한 이름을 얻을 수 있다.
                        max_index = np.where(J_list==max(J_list))[0][0]
                
                        #아래의 두개의 for문으로 각각 next_city, next_city_index를 구한다.
                        #이는 update_initial_pheromone메소드의 코드를 약간 수정하여 사용했다.
                        
                        for i, items in enumerate(candidate_distace_to.items()):
                            if items[1]*isEmpty[i] == distance[max_index]:
                                next_city = items[0]
                
                        for i ,city in enumerate(list_of_cities):
                            if city.name == next_city:
                                next_city_index = i
                              
                    else:
                        #rand>q0인 경우
                        
                        
                
                        #분자생성
                        numerator = []
                        for i in range(self.number_of_CL):
                            numerator.append((pheromone[i]**self.alpha)*( (1/distance[i])**self.beta )*(isEmpty[i]))
                
                        #분모생성     
                        denominator = sum( (np.array(pheromone)**self.alpha)*( (1/distance)**self.beta )*(isEmpty) ) 
                        
                        #분자와 분모의 누적합을 통하여 cumsum_list생성
                        cumsum_list = np.cumsum(numerator/denominator)
                
                        #cumsum-난수값을 통하여 어느도시를 선택할지 결정
                        TF_list_for_P = list(cumsum_list-np.random.rand()>0)
                        
                        #선택된 index저장.
                        selected_P_index = TF_list_for_P.index(True)
                
                        for i, items in enumerate(candidate_distace_to.items()):
                            if items[1]*isEmpty[i] == distance[selected_P_index]:
                                next_city = items[0]
                
                        for i ,city in enumerate(list_of_cities):
                            if city.name == next_city:
                                next_city_index = i
                            
        
                               
                    #다음도시가 선택됬으니 ant와 ant_length도 업데이트해준다.
                    ant.append(next_city)
                    ant_index.append(next_city_index)
                    ant_length.append(list_of_cities[ant_index[banbok]].distance_to[next_city])
                    
                    
                    #이제 다음도시를 구했으니 이 도시에 해당하는 total_candidate랑 candidate를 False로 만들어준다.
                    for city in list_of_cities:
                        city.total_candidate[next_city] = False
                    del list_of_cities[next_city_index].total_candidate[next_city]
                
                    for city in list_of_cities:
                        if next_city in city.candidate:
                            city.candidate[next_city] = False
                
                    
                
                #마지막도시와 첫번째출발도시의 length를 더해주기위해서 한줄을 추가해준다.
                ant_length.append(list_of_cities[next_city_index].distance_to[ant[0]])
                            
        
                
                #이제 ant와 ant_index에 저장된 도시의 순서를 이용하여으로 local페로몬 업데이트를 실행한다.
                #tsp의 경우 페로몬증발계수인 rho는 보통 0.1을 이용한다.

            
                # 앞으로 갈 때, 뒤로 갈 때
                # 모두 고려하여 각각 페로몬을 업데이트 시켜준다.
                for i in range(len(list_of_cities)):
                    list_of_cities[ant_index[i]].pheromone_to[ant[i-len(list_of_cities)+1]] = (1-self.rho)*list_of_cities[ant_index[i]].pheromone_to[ant[i-len(list_of_cities)+1]] + self.rho*self.tau0
                for i in range(len(list_of_cities)):    
                    list_of_cities[ant_index[len(list_of_cities)-1-i]].pheromone_to[ant[-i+len(list_of_cities)-2]] = (1-self.rho)*list_of_cities[ant_index[len(list_of_cities)-1-i]].pheromone_to[ant[-i+len(list_of_cities)-2]] + self.rho*self.tau0
                
                
                #이제 다음개미가 출발해야하므로
                #candidate, total_candidate를 다시 비워주자(empty),
                #즉, 다시 True값을 넣어준다.
                for pivot_city in list_of_cities:
                    for subordinate_city in list_of_cities:
                        pivot_city.total_candidate[subordinate_city.name] = True
                    del pivot_city.total_candidate[pivot_city.name]
                
                for pivot_city in list_of_cities:
                    for keys in pivot_city.candidate.keys():
                        pivot_city.candidate[keys] = True 
                    
                
                history_ant.append(ant)
                history_ant_index.append(ant_index)
                history_length.append(sum(ant_length))
                  

            
            #여기까지가 한마리의 개미가 도는것...
            #이제 이 코드를 개미의 마릿수 만큼 반복,,, 그 반복마다 local업데이트
            
            #그리고 모든 개미가 한바퀴씩 돌았다면, 그 개미들중에서 베스트 개미를 뽑는다.
            if epoch ==0:
                Initial_best_distance = min(history_length)
            
            if min(history_length) < best_length:
                best_index = history_length.index(min(history_length))
                best_ant = history_ant[best_index]
                best_ant_index = history_ant_index[best_index]
                best_length = min(history_length)
                
            self.history_of_bestLength.append(best_length)
            
            

            #베스트 솔루션에 해당하는 페로몬을 업데이트한다.(global update)
            for i in range(len(list_of_cities)):
                list_of_cities[best_ant_index[i]].pheromone_to[best_ant[i-len(list_of_cities)+1]] = (1-self.rho)*list_of_cities[best_ant_index[i]].pheromone_to[best_ant[i-len(list_of_cities)+1]] + self.rho*(1/best_length)
            for i in range(len(list_of_cities)):    
                list_of_cities[best_ant_index[len(list_of_cities)-1-i]].pheromone_to[best_ant[-i+len(list_of_cities)-2]] = (1-self.rho)*list_of_cities[best_ant_index[len(list_of_cities)-1-i]].pheromone_to[best_ant[-i+len(list_of_cities)-2]] + self.rho*(1/best_length)
            
            print('\nepochs :',epoch+1)
            print('best_ant :',best_ant)    
            print('best_distance :', best_length)
            
            
            
            
            
            
        # =============================================================================
        # =============================================================================
        # # #최종결과 & 시각화
        # =============================================================================
        # =============================================================================
        
        
        #결과 프린트
        print("\n\n###############")            
        print("### Result! ###")
        print("###############\n")
        print("final Route:", best_ant)
        print("Initial best distance:",Initial_best_distance)
        print("Final best distance:",best_length)
        
        
        
        #시각화
        plt.figure(figsize=(14,28))
        
        #시각화1. best_distacne값의 변화.
        plt.subplot(2,1,1)
        plt.title('Change of Best Distance according to epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Best Distance')
        plt.plot(range(self.total_epochs), self.history_of_bestLength, c='r', label="best Distance" )
        plt.legend(loc='upper right')
        plt.grid(True)
        
        #시각화2. 최종 best_route
        self.best_x=[]
        self.best_y=[]
        for i in range(len(list_of_cities)):
            self.best_x.append(list_of_cities[best_ant_index[i]].x)
            self.best_y.append(list_of_cities[best_ant_index[i]].y)
        self.best_x.append(list_of_cities[best_ant_index[0]].x)
        self.best_y.append(list_of_cities[best_ant_index[0]].y)

        plt.subplot(2,1,2)
        plt.title('Best Route')
        for i in range(len(list_of_cities)):
            plt.text(self.best_x[i]-np.mean(self.best_x)/40, self.best_y[i]+np.mean(self.best_y)/40, "{}".format(best_ant[i]), fontsize=10)    
        plt.plot(self.best_x,self.best_y,c="b",
                 lw=2, ls="-", marker="o", ms=10, mec="black", mew=1, mfc="white")
        
        














            
#실행
ACO(number_of_CL = 10,
    num_of_ants = None,
    alpha = 1,
    beta = 3,
    q0 = 0.9,
    rho = 0.1,
    total_epochs = 100)



