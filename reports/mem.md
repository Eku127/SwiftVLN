1. 历史采样分布。1.0 是均匀采样，>1.0 是偏向最近帧；2.0 属于中等偏近期

2. gtc：把所有历史 token 当成一个无序集合做全局聚类，固定输出 token 数

3. segment_gtc：先把历史切成 8 段，再每段做 GTC，最后按早到晚拼回去

4. 把 episode 第 1 帧作为初始观察放进 system prompt

5. 给每张图注入 pose 的 EMBEDDING

6. 给视觉每一张图 token 注入像素坐标 embedding