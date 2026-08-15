# 代码架构

> 框架页：本页面向贡献者，描述稳定模块边界和数据流；用户参数放在配置参考中。

## 1. 仓库结构

<!-- src/swiftvln、scripts、tools/s2r、baseline、environments。 -->

## 2. 训练链路

<!-- shell -> args -> dataset -> template -> model -> ms-swift。 -->

## 3. 评测链路

<!-- CLI -> runner -> evaluator/session -> backend -> results。 -->

## 4. Modeling

<!-- model registration、template、history、memory、embeddings。 -->

## 5. Backend 边界

<!-- base contract、factory、SatNav、Habitat。 -->

## 6. 配置与实验名称

<!-- experiment.py 为跨训练/评测的约束源。 -->

## 7. S2R 依赖方向

<!-- tools 可依赖核心 adapter 定义，核心运行时不依赖 tools。 -->

## 8. 持久化与分布式

<!-- train metadata、eval append log、rank marker、summary。 -->
