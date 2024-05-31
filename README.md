# Modelagem de Tópicos (Topic Modeling) em língua portuguesa PT-BR

![image](https://github.com/cristianweiss/topicmodeling/assets/32395788/f1de685f-da7a-4dd9-8aca-40030b2e0dd3)

CRISTIAN EDEL WEISS ¹ | email@cristianweiss.com

Mostramos como o uso da modelagem de tópicos, uma técnica de machine learning, pode ajudar os repórteres a examinar rapidamente grandes quantidades de texto a fim de detectar os assuntos (tópicos) mais relevantes abordados no corpus e, em seguida, a buscar os trechos desejados com exatidão para análise objetiva do conteúdo. Pode ser útil para examinar dossiês, relatórios, processos judiciais, debates eleitorais e livros. Vamos construir juntos o modelo de um algoritmo otimizado para lidar com textos em língua portuguesa, baseado nos métodos de modelagem de tópicos [Top2Vec](https://github.com/ddangelov/Top2Vec). Os participantes aprenderão a preparar os textos (pré-processamento), remover palavras irrelevantes, treinar o modelo e exibir os resultados em formas de tabelas e nuvem de palavras, úteis para publicação digital. Não é necessário conhecimento de programação ou Python, pois os usuários rodarão o script no Google Colab e precisarão apenas alterar variáveis que interessam ao próprio projeto.

**Apresentação**
Este notebook foi desenvolvido para participação no 19º Congresso Internacional de Jornalismo Investigativo em julho de 2024:
![image](https://github.com/cristianweiss/topicmodeling/assets/32395788/7cce3a67-aee5-4e1f-980c-6bddf4870fba)

Para saber mais sobre como funciona a modelagem de tópicos, acesse a apresentação:

👉 📊 : [Descubra pautas em textos longos](https://www.google.com/url?q=https%3A%2F%2Fwww.canva.com%2Fdesign%2FDAGFaA6xWog%2FzccpSdaH8qv5KnX_5mF6xg%2Fview)


**O QUE É MODELAGEM DE TÓPICOS**

É uma técnica de Processamento de Linguagem Natural (NLP, na sigla em inglês) e de aprendizagem de máquina que permite analisar grandes quantidades de texto de maneira automatizada a partir de pistas geradas pelo algoritmo, que indica possíveis assuntos tratados ou mencionados no texto. Basicamente, o algoritmo transforma cada palavra em números (vetores) e, com isso, calcula o grau de proximidade dessas palavras no texto, gerando agrupamentos (clusters) de termos considerados relevantes no corpus textual.

**POR QUE É ÚTIL EM REPORTAGEM, INVESTIGAÇÃO OU PESQUISA**

A técnica funciona bem em situações em que você precisa lidar com grandes quantidades de texto e cujo teor é inicialmente desconhecido e precisa ser explorado, mas não há tempo ou é humanamente impossível (caso de grandes bases de dados) ler manualmente cada entrada de texto. Pode ser útil em casos como:

Análise prévia de dossiês, peças processuais, inquéritos policiais e da promotoria pública, atas, discursos, livros.
Em cobertura eleitoral, pode ser útil para analisar planos de governo, debates, discursos, longas entrevistas, conjunto de tweets.
Caso a reportagem precise analisar uma grande base de dados que contém uma das colunas como um campo textual, como a descrição de infrações ambientais, por exemplo, é útil para gerar pistas dos casos mais "quentes" a se analisar manualmente.
COMO FUNCIONA

Existem várias ténicas de modelagem de tópicos em Python e R. Cada uma com suas vantagens e desvantagens. Aqui, vamos utilizar a técnica Top2Vec (Angelov, 2020), uma das mais modernas, em Python, e que funciona bem em língua portuguesa.

O algoritmo transforma as palavras em números (vetores) e calcula o grau de proximidade delas no texto, gerando agrupamentos (clusters) de termos considerados relevantes no corpus.

Ao incorporar documentos e palavras no espaço vetorial, o algoritmo tenta encontrar grupos densos de documentos e, em seguida, identificar quais palavras atraíram esses documentos.

Cada área densa identificada é um tópico. As palavras que atraíram os documentos para a área densa são as palavras destacadas no tópico.

**PASSO A PASSO**

![image](https://github.com/cristianweiss/topicmodeling/assets/32395788/3dccaaa9-9077-4856-b0b5-b99230791698)

**USE EM SEUS PROJETOS**

❗ ***Toda vez que você se deparar com esse sinal, significa que é um campo que precisa ser adaptado para o seu projeto, como um link personalizado para seu arquivo pdf ou video.***

✍ **PODERIA DAR CRÉDITOS?**

Estes modelos foram adaptados de forma didática para iniciantes em Python ou pessoas sem experiência com programação e otimizado para lidar com textos em língua portuguesa. Espero que seja útil para os seus projetos. Se você se basear nestes modelos para seus projetos acadêmicos ou jornalísticos, ficarei feliz e grato se citá-lo ou creditá-lo como referência 🤗. Dúvidas, ideias, aprimoramentos ou sugestões são bem-vindas e podem ser enviadas para email@cristianweiss.com ou [cristianweiss.com](https://cristianweiss.com/).

¹ Jornalista com especialização em reportagem guiada por dados (Deutsche Welle Akademie), mestre em Mídia, Tecnologia e Sociedade (M.Sc., Universidade de Darmstadt, Alemanha) e doutorando em Ciência Política pela Universidade de Mannheim, Alemanha. Atuou como repórter e editor de jornais e portais do Grupo RBS e NSC Comunicação, no Sul do Brasil, e na Deutsche Welle, na Alemanha. Integrou o projeto Comprova entre 2018 e 2021. É membro da Abraji e da Jeduca.

# Referências


**TESE DE MESTRADO QUE UTILIZOU MODELAGEM DE TÓPICOS SOBRE A LAI EM LÍNGUA PORTUGUESA**

> WEISS, Cristian Edel. Transparency in education in the Bolsonaro government: a topic modeling of requests via the Brazilian Freedom of Information Act from the perspective of Watchdog Journalism. 2023. 181 f. Master's Thesis - Darmstadt University of Applied Sciences, Darmstadt, Germany, 2023.

**APRESENTAÇÃO NO CONGRESSO DA ABRAJI 2024**

> WEISS, Cristian Edel. Descubra pautas em textos longos. São Paulo: [s. n.], 2024. Disponível em: https://www.canva.com/design/DAGFaA6xWog/zccpSdaH8qv5KnX_5mF6xg/view. Acesso em: 27 maio 2024.

**DOCUMENTAÇÃO DA BIBLIOTECA TOP2VEC**

> ANGELOV, Dimo. Top2Vec: Distributed Representations of Topics. [S. l.]: arXiv, 2020. Disponível em: http://arxiv.org/abs/2008.09470. Acesso em: 27 maio 2024.

> Top2Vec — Top2Vec 1.0.29 documentation. (n.d.). https://top2vec.readthedocs.io/en/latest/Top2Vec.html


**FONTE DOS DADOS UTILIZADOS NESTE NOTEBOOK**

> TSE. Divulgação de Candidaturas e Contas Eleitorais. [S. l.], 2022. Disponível em: https://divulgacandcontas.tse.jus.br/divulga/#/. Acesso em: 27 maio 2024.

> DEBATE NA BAND: PRESIDENCIAL 2022. [S. l.: s. n.], 2022. Disponível em: https://www.youtube.com/watch?v=WwdgWl_nmKI. Acesso em: 27 maio 2024.

> DATA FIXERS. Data Fixers e Fiquem Sabendo: Multas do Ibama em 2022. Fonte: sistema “Consulta de Autuações Ambientais e Embargos”, do Ibama. , 2023. Disponível em: https://docs.google.com/spreadsheets/d/1CO33_gmQ_6s7zEhMR8mojdDnDTZ7EanWysqov20qlBY/edit?usp=sharing&usp=embed_facebook. Acesso em: 27 maio 2024.Google Spreadsheets

> FIQUEM SABENDO. Data Fixers: As maiores multas ambientais de 2022 no Brasil. [S. l.], 2023. Disponível em: https://fiquemsabendo.com.br/meio-ambiente/data-fixers-as-maiores-multas-ambientais-de-2022-no-brasil. Acesso em: 30 maio 2024.
