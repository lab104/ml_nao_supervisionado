# Instituto de Educação Superior de Brası́lia – IESB
## Pós-Graduação em Inteligência Artificial
### Disciplina de Computação Cognitiva 3 / Turma 2021-1
#### Trabalho Final - Análise de Sentimento

                          EQUIPE:
                          - LUCAS DE SOUSA BRITO, MAT:2186330019, TURMA: 2021-1
                          - PABLO NOGUEIRA OLIVEIRA, MAT:2186330027, TURMA: 2021-1
                          - MATHEUS BARBOSA OLIVEIRA, MAT:2186330037, TURMA: 2021-1
                          
                          
# INTRODUÇÃO

O presente exercício trata da elaboração de modelos de análise de sentimento 
na base [Sentiment140](http://help.sentiment140.com/for-students) que contém 
1,6 milhões de tweets cujo sentimento (positivo, neutro ou negativo) foram 
classificados pelo autor da base.

Segundo [Go, Bhayani e Huang (2013)](https://cs.stanford.edu/people/alecmgo/papers/TwitterDistantSupervision09.pdf), 
autores da base Sentiment140, classificaram o sentimento baseado em emoticons
existentes nos tweets de forma automática.

A chamada para este exercício foi a de montar diversos modelos de análise 
de sentimento e comparar os indicadores de performance da detecção para que 
os alunos percebam a diferença dos resultados quando usando as técnicas 
de deep learning, bem como, discursem sobre a estruturação do modelo 
e os fundamentos das técnicas usadas.

# ESTRUTURA GERAL DOS NOTBOOKS

Para os notebooks foram efetuadas:
* Carga da base 
* Limpeza dos tweets
* Tokenização 
   * Criação de dicionário de palavras 
   * Transformação das expressões em vetores numéricos
   * Padding dos vetores para ter o mesmo tamanho
* Treinamento
* Apresentação da matriz de confusão

# RESULTADO GERAL

* Deep learning ~ 74,6%
* Regressão Logística ~ 70%
* Bagging com árvores de decisão ~ 67,4%
* Multinomial NB ~ 69,4%


