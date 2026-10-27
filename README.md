# ai-harness

Fundação independente e instalável para um runtime de agentes de IA, com
controle de ciclo de vida, integrações desacopladas de fornecedores e decisões
de política fora do modelo. O projeto está em estágio inicial (`0.1.0`) e inclui
providers locais e agentes read-only; conectores opcionais precisam ser
configurados e verificados pela aplicação que os utiliza.

## Requisitos

- Python 3.12 ou superior (limitado a versões anteriores a 3.16)
- Git

Importar o núcleo e executar os testes não exige conta, chave ou serviço. Uma
execução real usa o Ollama local configurado; o provider Anthropic opcional
requer uma chave de API própria.

## Instalação no Windows

Na pasta do projeto:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test,mcp]"
python -m pytest
```

Para instalar a partir de outro projeto Python local, use
`python -m pip install <caminho-para-ai-harness>`. O pacote também pode ser
construído e versionado como distribuição Python.

## Primeira execução pela CLI

Após ativar o ambiente e instalar o pacote em modo editável:

```powershell
harness run "Analise a estrutura deste projeto"
```

No Windows PowerShell, se o comando `harness` ainda não estiver disponível,
use o módulo diretamente:

```powershell
python -m ai_harness run "Resuma este pedido em uma frase"
```

O provider padrão é Ollama local (`AI_HARNESS_PROVIDER=ollama`), usando o
modelo `qwen3.8:27b` por padrão. O provider e o modelo podem ser selecionados
por `AI_HARNESS_PROVIDER`, `AI_HARNESS_MODEL` e `AI_HARNESS_OLLAMA_URL`.
Falhas de rede, autenticação ou geração são retornadas como falhas; não há
fallback silencioso para simulação. Para executar explicitamente o provider
determinístico de desenvolvimento, use `--provider simulated`.

Exemplo de saída:

```text
Mode: real provider=ollama model=qwen3.8:27b.
History database: C:\Users\example\AppData\Local\ai-harness\history.sqlite3
[10:15:22] execution.queued execution=<id> status=queued
[10:15:22] execution.started execution=<id> status=running
[10:15:22] agent.iteration.started execution=<id> status=running Iteration 1/8
[10:15:22] provider.response execution=<id> status=running Provider mode: real
[10:15:22] execution.completed execution=<id> status=completed Mode: real
Execution ID: <id>
State: completed
Result: <texto gerado pelo modelo>
```

Comandos disponíveis:

```powershell
harness --provider simulated run "Teste determinístico"
harness status <execution_id>
harness history
harness tools list
```

## Maestro e agentes com provider real

O comando `maestro run` usa o provider configurado para propor um plano
estruturado, valida tarefas e dependências e o persiste aguardando aprovação.
Os agentes `analyst` e `reviewer` leem somente arquivos de texto abaixo de
`AI_HARNESS_ALLOWED_ROOTS`, com limites de quantidade/tamanho e sem execução de
comandos ou escrita. O analyst produz fatos, hipóteses e recomendações com
citações de evidência; o reviewer recebe os resultados dependentes e deve
identificar achados independentes. Resultados, estados e eventos permanecem no
SQLite do Maestro.

O provider Anthropic é opcional: instale `.[anthropic]`, configure
`AI_HARNESS_PROVIDER=anthropic`, `AI_HARNESS_MODEL` e `ANTHROPIC_API_KEY`. O
login do Claude Code não autentica a API Anthropic do harness.

No PowerShell, as operações são:

```powershell
# Criar e persistir o plano, sem iniciar os agentes
python -m ai_harness --db .\maestro.sqlite3 maestro run "Analise esta solicitação"

# Aprovar o plano pelo run_id mostrado pelo comando anterior
python -m ai_harness --db .\maestro.sqlite3 maestro approve <run_id>

# Criar o plano e aprová-lo imediatamente
python -m ai_harness --db .\maestro.sqlite3 maestro run --approve "Analise esta solicitação"

python -m ai_harness --db .\maestro.sqlite3 maestro status <run_id>
python -m ai_harness --db .\maestro.sqlite3 maestro plan <run_id>
python -m ai_harness --db .\maestro.sqlite3 maestro tasks <run_id>
python -m ai_harness --db .\maestro.sqlite3 maestro tasks <run_id> --agent analyst
python -m ai_harness --db .\maestro.sqlite3 maestro messages <run_id>
python -m ai_harness --db .\maestro.sqlite3 maestro history
python -m ai_harness --db .\maestro.sqlite3 maestro approvals
python -m ai_harness --db .\maestro.sqlite3 maestro agents
python -m ai_harness --db .\maestro.sqlite3 maestro cancel <run_id>
```

`maestro run` sem `--approve` apenas prepara o plano. `approve` aceita um plano
pendente; `approve --reject` o rejeita. O banco compartilhado com o runtime
preserva plano, estado, tarefas, mensagens, resultados e eventos para consulta e
para aprovar uma execução pendente após reiniciar o processo. A solicitação de
cancelamento pode ser emitida por outro processo enquanto a execução está ativa.
O fluxo atual não retoma tarefas parcialmente executadas nem repete
automaticamente falhas.

Agentes simulados adicionais podem ser descritos em JSON, passando
`--provider simulated --agents-config <arquivo>` antes de `maestro`. As
ferramentas simuladas não são habilitadas para os agentes reais.

Exemplo de configuração (salve como `.\agents.json` e forneça também ao aprovar
um plano criado com essa configuração):

```json
{
  "agents": [
    {
      "agent_id": "analysis-local",
      "name": "Local Analyst",
      "description": "Simulated analysis specialist.",
      "specialty": "analysis",
      "provider": "simulated",
      "model": "deterministic",
      "capabilities": ["analysis"],
      "authorized_tools": ["echo"],
      "permissions": ["runtime.read"],
      "enabled": true
    },
    {
      "agent_id": "review-local",
      "name": "Local Reviewer",
      "description": "Independent simulated reviewer.",
      "specialty": "review",
      "provider": "simulated",
      "model": "deterministic",
      "capabilities": ["review"],
      "authorized_tools": ["echo"],
      "permissions": ["runtime.read"],
      "enabled": true
    }
  ]
}
```

```powershell
python -m ai_harness --db .\maestro.sqlite3 --provider simulated --agents-config .\agents.json maestro run "Solicitação"
python -m ai_harness --db .\maestro.sqlite3 --provider simulated --agents-config .\agents.json maestro approve <run_id>
```

Ao usar `Maestro` diretamente, forneça um registro de agentes e um planejador
provider-backed. O modo determinístico só é ativado com `simulation=True`
(ou pela CLI `--provider simulated`).

Não há uma rede de serviços para iniciar. A CLI é local e não expõe um painel
ou servidor de monitoramento; o acompanhamento disponível é pelo SQLite e pelos
comandos acima.

O banco SQLite fica em `%LOCALAPPDATA%\ai-harness\history.sqlite3` no Windows.
É possível informar outro caminho com `harness --db .\local.sqlite3 run "tarefa"`
(a opção global vem antes do subcomando). `status` consulta o banco; o
histórico e os eventos permanecem disponíveis após fechar e reabrir o processo.
Use `status` para consultar detalhes de eventos, chamadas de ferramentas e
aprovações.

Para integração como biblioteca, `Harness` recebe um `ModelProvider`,
`ToolRegistry`, política e `SQLiteExecutionStore`. O provider só pode propor
chamadas; o runtime valida os argumentos, exige que a ferramenta esteja
registrada e aplica `ToolPolicy` antes de invocá-la. Use `approval_handler`
para fornecer uma confirmação humana confiável para ferramentas de escrita ou
ações externas. Sem essa confirmação, o estado fica `waiting_approval` e a
ferramenta não é executada. A interface de aprovação é de biblioteca, não há
comando CLI para retomar execuções pendentes.

O contrato legado de ciclo de vida em `Execution` e as aprovações das
integrações existentes permanecem disponíveis. Consulte
[`docs/INTEGRATIONS.md`](docs/INTEGRATIONS.md) para contratos e criação de
adaptadores.

## O que está implementado

- Contexto de execução com UUID e metadados validados por Pydantic.
- Estados explícitos e transições validadas; cada transição gera um evento
  estruturado com timestamp UTC.
- Contrato assíncrono `ExecutionStore`, sem acoplar o núcleo a um banco.
- Classificação inicial de risco e decisões aprovadas, pendentes ou rejeitadas
  para ações de leitura, escrita e externas; comprovantes de escrita/efeitos
  externos são assinados em memória e vinculados à execução e solicitação exatas.
- Harness assíncrono de tarefas com limite de iterações, timeout, cancelamento,
  validação de ferramenta e eventos estruturados emitidos em tempo real.
- Providers reais Ollama e Anthropic opcional, além do provider simulado
  selecionado explicitamente para desenvolvimento.
- Agentes read-only `analyst` e `reviewer`, planejamento estruturado e servidor
  MCP stdio com execução persistida.
- SQLite para execuções, eventos, chamadas, resumos de resultado e decisões de
  aprovação; argumentos não são gravados em eventos nem em chamadas.
- Contratos para agentes, transportes MCP e plataformas externas, registro de
  integrações/skills e políticas existentes.

## Integração MCP

Instale a dependência opcional `.[mcp]` e registre o comando stdio
`python -m ai_harness.mcp_server` no Claude Code. Consulte
[`docs/AI_HARNESS_MCP.md`](docs/AI_HARNESS_MCP.md) para escopo, allowlist,
ferramentas, validação e configuração no Windows.

O servidor MCP não oferece ferramentas de escrita, shell ou rede aos agentes.
Planos gerados ficam aguardando aprovação; a chamada explícita a
`harness_run_agent` executa apenas análise/revisão read-only solicitada pelo
chamador. Não há retentativa automática nem retomada de tarefas parcialmente
executadas. O banco persiste solicitações e resultados; não inclua nele dados
que não devam ficar em disco. A configuração de autenticação Anthropic e o
acesso ao modelo local continuam independentes da sessão Claude Code.

## Licenciamento

O código original deste projeto está sob MIT; a decisão e suas implicações
estão registradas em [`docs/LICENSING.md`](docs/LICENSING.md). Isso não altera
nem substitui as licenças de dependências, modelos, serviços ou outros
componentes de terceiros.
