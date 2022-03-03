# line graph machting
1. convert graph to line graph ***G***. Then we get line graph ***G_s*** and ***G_t***. 
2. construct line graph permutaion matrix **Perm**.
2. update ***G_s*** and ***G_t*** with MPNN.
3. get top-k points ***P_s*** from ***G_s***, and ***P_t*** from ***G_t***.
4. matching ***P_s*** and ***P_t***, return gradients.
