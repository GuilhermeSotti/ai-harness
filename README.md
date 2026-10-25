# ai-harness

Fundação independente e instalável para um runtime de agentes de IA, com
controle de ciclo de vida, integrações desacopladas de fornecedores e decisões
de política fora do modelo. O projeto está em estágio inicial (`0.1.0`); os
adaptadores de provedores são contratos e precisam ser implementados e
verificados pela aplicação que os utiliza.

## Requisitos

- Python 3.12 ou superior (limitado a versões anteriores a 3.16)
- Git

O projeto não exige contas, chaves de API, serviços pagos ou acesso à rede para
usar o núcleo e as políticas desta versão. Nenhum serviço externo é conectado
automaticamente.

## Instalação no Windows

Na pasta do projeto:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python -m pytest
```

Para instalar a partir de outro projeto Python local, use
`python -m pip install <caminho-para-ai-harness>`. O pacote também pode ser
construído e versionado como distribuição Python.

## Primeira execução pela CLI

Após ativar o ambiente e instalar o pacote em modo editável:

```powershell
harness run "Conte as palavras desta tarefa"
```

No Windows PowerShell, se o comando `harness` ainda não estiver disponível,
use o módulo diretamente:

```powershell
python -m ai_harness run "Conte as palavras desta tarefa"
```

A implementação inicial é **somente simulada**. A saída informa explicitamente
que não usou modelo real nem API; o provider simulado propõe uma chamada à
ferramenta `echo`, autorizada e sem efeitos externos, e então produz um texto
marcado como simulado. Não há provider real incluído. A interface
`ModelProvider` permite adaptadores futuros, mas o runtime não conecta
provedores automaticamente.

Exemplo de saída:

```text
Mode: simulated (no model or external API was called).
History database: C:\Users\example\AppData\Local\ai-harness\history.sqlite3
[10:15:22] execution.queued execution=<id> status=queued
[10:15:22] execution.started execution=<id> status=running
[10:15:22] agent.iteration.started execution=<id> status=running Iteration 1/8
[10:15:22] provider.response execution=<id> status=running Provider mode: simulated
[10:15:22] tool.started execution=<id> status=running tool=echo
[10:15:22] tool.completed execution=<id> status=running tool=echo ...
[10:15:22] execution.completed execution=<id> status=completed Mode: simulated
Execution ID: <id>
State: completed
Result: Simulated run completed; no real model was used.
```

Comandos disponíveis:

```powershell
harness status <execution_id>
harness history
harness tools list
```

## Maestro e coordenação simulada

O comando `maestro` cria um plano persistido que aguarda aprovação antes de
delegar. O perfil padrão encaminha tarefas independentes a dois agentes locais
(`analyst` e `reviewer`). Ambos executam pelo `Harness` existente e pelo
`SimulatedProvider`; não há conexão de modelos nem rede. O Maestro não recebe
ferramentas próprias, e cada agente só recebe as ferramentas explicitamente
autorizadas em seu registro e filtradas pela política do runtime.

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
`--agents-config <arquivo>` antes de `maestro`. O arquivo aceita somente agentes
com `provider` igual a `simulated`; provedores externos são recusados até que
um adaptador real seja implementado e verificado. As permissões de ferramenta
requerem tanto o nome da ferramenta em `authorized_tools` quanto
`runtime.read` para leitura ou `tool:<nome>:<ação>` para uma permissão
específica. A política do runtime continua exigindo aprovação exata para
escritas e ações externas.

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
python -m ai_harness --db .\maestro.sqlite3 --agents-config .\agents.json maestro run "Solicitação"
python -m ai_harness --db .\maestro.sqlite3 --agents-config .\agents.json maestro approve <run_id>
```

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
- Contrato neutro de provider, agente inicial e ferramenta `echo` exclusivamente
  simulados, sem chamadas de rede.
- SQLite para execuções, eventos, chamadas, resumos de resultado e decisões de
  aprovação; argumentos não são gravados em eventos nem em chamadas.
- Contratos para agentes, transportes MCP e plataformas externas, registro de
  integrações/skills e políticas existentes.

## Limites desta entrega

Não há provider de modelo real, APIs, conectores externos, execução de comandos
ou retomada de execução pendente pela CLI. Não há retries automáticos; handlers
de ferramentas devem ser assíncronos para que timeout e cancelamento sejam
aplicados. Providers, handlers e aprovações reais precisam ser integrados
conscientemente por uma aplicação confiável.
Confirme identidade e autorização de quem aprova fora do harness. O banco
persiste o texto da tarefa; não inclua nele credenciais ou dados que não devam
ficar em disco. Eventos e registros de chamadas guardam resumos e digest dos
argumentos, e erros são reduzidos ao tipo da exceção para evitar vazar
conteúdo. Esta versão é um núcleo local inicial, não um serviço de produção.

## Licenciamento

O código original deste projeto está sob MIT; a decisão e suas implicações
estão registradas em [`docs/LICENSING.md`](docs/LICENSING.md). Isso não altera
nem substitui as licenças de dependências, modelos, serviços ou outros
componentes de terceiros.
